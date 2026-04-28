#!/usr/bin/env python3
"""读取 Piper 数据集，使用 MeshCat 在网页中回放 IK 机械臂轨迹。"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import meshcat.geometry as g
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from piper_ik_visualizer import html_response, json_response, make_meshcat_viewer, make_server, render_html_template
from piper_dataset_reader import (
    DEFAULT_DATASET,
    DEFAULT_GRIPPER_COLUMN,
    DEFAULT_POSE_COLUMN,
    ROOT,
    PiperDatasetReader,
    ReplayConfig,
    build_pose_plan,
    gripper_sequence_for_poses,
)
from piper_ik_utils import (
    DEFAULT_INITIAL_JOINTS,
    DEFAULT_JOINTS,
    DEFAULT_URDF,
    PinocchioIK,
    apply_gripper_values_to_qs,
    joint_vector_from_names,
    solve_ik_sequence,
)


def make_transform(translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = translation
    return transform


def make_path_line(points: np.ndarray) -> g.Line:
    colors = np.zeros((3, points.shape[0]), dtype=np.float64)
    if points.shape[0] > 1:
        progress = np.linspace(0.0, 1.0, points.shape[0])
    else:
        progress = np.array([1.0])
    colors[0, :] = 0.07 + 0.02 * progress
    colors[1, :] = 0.42 + 0.28 * progress
    colors[2, :] = 0.30 + 0.10 * progress
    return g.Line(
        g.PointsGeometry(points.T, color=colors),
        g.LineBasicMaterial(vertexColors=True, linewidth=5),
    )


def make_target_path_line(points: np.ndarray) -> g.Line:
    return g.Line(
        g.PointsGeometry(points.T),
        g.LineBasicMaterial(color=0xC2185B, linewidth=4, transparent=True, opacity=0.72),
    )


def sample_indices(length: int, count: int) -> np.ndarray:
    if count <= 0 or length <= 0:
        return np.asarray([], dtype=int)
    return np.unique(np.linspace(0, length - 1, min(count, length)).round().astype(int))


class DatasetWeb3DApp:
    def __init__(
        self,
        *,
        parquet_file: Path,
        dataset_root: Path,
        pose_column: str,
        gripper_column: str | None,
        urdf_path: Path,
        package_dir: Path,
        ee_frame: str,
        config: ReplayConfig,
        initial_joint_values: np.ndarray,
        episode_index: int | None,
        invert_gripper: bool,
        ik_options: dict[str, Any] | None,
        meshcat_port: int,
        ghost_count: int,
        marker_count: int,
        root_node: str,
    ) -> None:
        self.lock = threading.RLock()
        self.parquet_file = parquet_file
        self.dataset_root = dataset_root
        self.pose_column = pose_column
        self.gripper_column = gripper_column
        self.ee_frame = ee_frame
        self.config = config
        self.initial_joint_values = np.asarray(initial_joint_values, dtype=np.float64).reshape(-1)
        self.episode_index = episode_index
        self.invert_gripper = invert_gripper
        self.ik_options = dict(ik_options or {})
        self.root_node = root_node
        self.ghost_count = ghost_count
        self.marker_count = marker_count
        self.dataset_reader = PiperDatasetReader(
            dataset_root,
            pose_column=pose_column,
            gripper_column=gripper_column,
        )
        self.fps = self.dataset_reader.fps()
        try:
            info = self.dataset_reader.load_info()
            self.total_episodes = int(info["total_episodes"]) if info.get("total_episodes") is not None else None
        except Exception:
            self.total_episodes = None
        self.ik = PinocchioIK(
            urdf_path=urdf_path,
            package_dir=package_dir,
            ee_frame=ee_frame,
            active_joint_names=DEFAULT_JOINTS,
            initial_joint_values=self.initial_joint_values,
        )
        self.deltas = np.empty((0, 6), dtype=np.float64)
        self.gripper: np.ndarray | None = None
        self.gripper_frames: np.ndarray | None = None
        self.poses = np.empty((0, 6), dtype=np.float64)
        self.qs = np.empty((0, self.ik.model.nq), dtype=np.float64)
        self.ik_results = []
        self.target_points = np.empty((0, 3), dtype=np.float64)
        self.ee_points = np.empty((0, 3), dtype=np.float64)
        self.frame = 0
        self.viewer, self.meshcat_server = make_meshcat_viewer(meshcat_port)
        self.meshcat_url = self.viewer.url()
        self.viz = MeshcatVisualizer(self.ik.model, self.ik.collision_model, self.ik.visual_model)
        self.viz.initViewer(viewer=self.viewer, open=False)
        self.viz.loadViewerModel(rootNodeName=f"{root_node}/current")
        self.ghost_viz: list[MeshcatVisualizer] = []
        self.load_parquet(parquet_file, episode_index)

    def _compute_ee_points(self) -> np.ndarray:
        points = []
        data = self.ik.model.createData()
        for q in self.qs:
            pin.forwardKinematics(self.ik.model, data, q)
            pin.updateFramePlacements(self.ik.model, data)
            points.append(data.oMf[self.ik.ee_frame_id].translation.copy())
        return np.asarray(points, dtype=np.float64)

    def _load_static_scene(self, marker_count: int, ghost_count: int) -> None:
        self.viewer[f"{self.root_node}/target_path"].set_object(make_target_path_line(self.target_points))
        self.viewer[f"{self.root_node}/ee_path"].set_object(make_path_line(self.ee_points))
        for marker_id, index in enumerate(sample_indices(len(self.ee_points), marker_count)):
            self.viewer[f"{self.root_node}/markers/{marker_id:03d}"].set_object(
                g.Sphere(0.009),
                g.MeshLambertMaterial(color=0x1f8a5b, transparent=True, opacity=0.65),
            )
            self.viewer[f"{self.root_node}/markers/{marker_id:03d}"].set_transform(make_transform(self.ee_points[index]))
        for ghost_id, index in enumerate(sample_indices(len(self.qs), ghost_count)):
            ghost = MeshcatVisualizer(self.ik.model, None, self.ik.visual_model)
            ghost.initViewer(viewer=self.viewer, open=False)
            alpha = 0.10 + 0.25 * ((ghost_id + 1) / max(ghost_count, 1))
            ghost.loadViewerModel(
                rootNodeName=f"{self.root_node}/ghosts/{ghost_id:03d}",
                visual_color=[0.18, 0.46, 0.70, alpha],
            )
            ghost.display(self.qs[index])
            self.ghost_viz.append(ghost)
        self.viewer[f"{self.root_node}/current_tip"].set_object(
            g.Sphere(0.014),
            g.MeshLambertMaterial(color=0xd12f2f, transparent=False, opacity=1.0),
        )

    def _clear_replay_scene(self) -> None:
        self.viewer[f"{self.root_node}/target_path"].delete()
        self.viewer[f"{self.root_node}/ee_path"].delete()
        self.viewer[f"{self.root_node}/markers"].delete()
        self.viewer[f"{self.root_node}/ghosts"].delete()
        self.viewer[f"{self.root_node}/current_tip"].delete()
        self.ghost_viz = []

    def _reset_initial_configuration(self) -> np.ndarray:
        self.ik.q = joint_vector_from_names(self.ik.model, DEFAULT_JOINTS, self.initial_joint_values)
        return self.ik.forward_pose6()

    def load_parquet(self, parquet_file: Path, episode_index: int | None) -> dict[str, Any]:
        with self.lock:
            self.parquet_file = parquet_file
            self.episode_index = episode_index
            self.deltas = self.dataset_reader.load_pose_deltas(parquet_file)
            self.gripper = (
                self.dataset_reader.load_gripper_values(parquet_file, len(self.deltas))
                if self.gripper_column
                else None
            )
            self.gripper_frames = gripper_sequence_for_poses(
                gripper_values=self.gripper,
                total_steps=len(self.deltas),
                start=self.config.start,
                end=self.config.end,
            )
            start_pose = self._reset_initial_configuration()
            self.poses = build_pose_plan(self.deltas, start_pose, self.config)
            self.target_points = self.poses[:, :3].copy()
            self.qs, self.ik_results = solve_ik_sequence(self.ik, self.poses, ik_options=self.ik_options)
            self.qs = apply_gripper_values_to_qs(
                model=self.ik.model,
                qs=self.qs,
                gripper_values=self.gripper_frames,
                invert=self.invert_gripper,
            )
            self.ee_points = self._compute_ee_points()
            self.frame = 0
            self._clear_replay_scene()
            self._load_static_scene(marker_count=self.marker_count, ghost_count=self.ghost_count)
            return self.display_frame(0)

    def load_episode(self, episode_index: int) -> dict[str, Any]:
        return self.load_parquet(self.dataset_reader.episode_parquet_path(episode_index), episode_index)

    def display_frame(self, frame: int) -> dict[str, Any]:
        with self.lock:
            frame = int(max(0, min(frame, len(self.qs) - 1)))
            self.frame = frame
            self.viz.display(self.qs[frame])
            self.viewer[f"{self.root_node}/current_tip"].set_transform(make_transform(self.ee_points[frame]))
            return self.state()

    def state(self) -> dict[str, Any]:
        result = self.ik_results[self.frame]
        pose = self.poses[self.frame]
        gripper_value = None if self.gripper_frames is None else float(self.gripper_frames[self.frame])
        return {
            "file": str(self.parquet_file),
            "episode_index": self.episode_index,
            "total_episodes": self.total_episodes,
            "meshcat_url": self.meshcat_url,
            "frame": self.frame,
            "frame_count": len(self.qs),
            "fps": self.fps,
            "delta_mode": self.config.delta_mode,
            "pos_scale": self.config.pos_scale,
            "rot_scale": self.config.rot_scale,
            "z_lift": self.config.z_lift,
            "initial_x_offset": self.config.initial_x_offset,
            "initial_y_offset": self.config.initial_y_offset,
            "initial_z_offset": self.config.initial_z_offset,
            "initial_joints": [float(value) for value in self.initial_joint_values],
            "gripper_value": gripper_value,
            "invert_gripper": self.invert_gripper,
            "converged_count": int(sum(item.converged for item in self.ik_results)),
            "position_error": float(result.position_error),
            "rotation_error": float(result.rotation_error),
            "converged": bool(result.converged),
            "pose": [float(value) for value in pose],
        }


class Web3DHandler(BaseHTTPRequestHandler):
    app: DatasetWeb3DApp

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            html_response(self, index_html(self.app.meshcat_url))
        elif path == "/api/state":
            json_response(self, {"ok": True, "state": self.app.state()})
        else:
            json_response(self, {"ok": False, "error": "Not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/frame":
                state = self.app.display_frame(int(payload.get("frame", 0)))
                json_response(self, {"ok": True, "state": state})
            elif path == "/api/episode":
                state = self.app.load_episode(int(payload.get("episode_index", 0)))
                json_response(self, {"ok": True, "state": state})
            else:
                json_response(self, {"ok": False, "error": "Not found"}, 404)
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc)}, 400)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def index_html(meshcat_url: str) -> str:
    return render_html_template("replay_piper_dataset_web3d.html", meshcat_url=meshcat_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper dataset replay 3D web viewer.")
    parser.add_argument("--file", "-f", type=Path, default=None, help="直接指定 episode parquet 文件；不能和 --episode-index 同时使用")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET, help="数据集根目录")
    parser.add_argument("--episode-index", type=int, default=None, help="按数据集 episode 编号选择 parquet；不填时默认 episode 0")
    parser.add_argument("--pose-column", default=DEFAULT_POSE_COLUMN, help="相对末端位姿列")
    parser.add_argument("--gripper-column", default=DEFAULT_GRIPPER_COLUMN, help="夹爪列，留空则不读")
    parser.add_argument("--invert-gripper", action="store_true", help="反向显示夹爪开合，适用于数据集 1=闭合 的情况")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="URDF 路径")
    parser.add_argument("--package-dir", type=Path, default=ROOT, help="包含 piper_description 包的目录")
    parser.add_argument("--ee-frame", default="gripper_base", help="IK 末端 frame/link")
    parser.add_argument("--delta-mode", choices=("incremental", "from-start"), default="incremental", help="相对位姿解释方式")
    parser.add_argument("--start", type=int, default=0, help="开始帧")
    parser.add_argument("--end", type=int, default=None, help="结束帧")
    parser.add_argument("--pos-scale", type=float, default=0.6, help="位置增量缩放")
    parser.add_argument("--rot-scale", type=float, default=1.0, help="姿态增量缩放")
    parser.add_argument(
        "--initial-joints",
        type=float,
        nargs=6,
        default=DEFAULT_INITIAL_JOINTS,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="离线 IK 初始关节姿态，默认: 0.000 0.004 -0.281 -0.000 0.364 0.000",
    )
    parser.add_argument("--z-lift", type=float, default=0.0, help="离线初始 z 抬升")
    parser.add_argument("--initial-x-offset", type=float, default=0.0, help="初始位姿 x 偏移，米；向后 20cm 可设为 -0.2")
    parser.add_argument("--initial-y-offset", type=float, default=0.0, help="初始位姿 y 偏移，米")
    parser.add_argument("--initial-z-offset", type=float, default=0.0, help="初始位姿 z 偏移，米")
    parser.add_argument("--position-cost", type=float, default=1.0, help="pink 末端位置任务代价")
    parser.add_argument("--orientation-cost", type=float, default=0.5, help="pink 末端姿态任务代价")
    parser.add_argument("--posture-cost", type=float, default=1e-3, help="pink 姿态保持正则代价")
    parser.add_argument("--ik-pos-tol", type=float, default=5e-4, help="pink 位置收敛阈值，米")
    parser.add_argument("--ik-rot-tol", type=float, default=5e-3, help="pink 旋转收敛阈值，弧度")
    parser.add_argument("--ik-max-iters", type=int, default=40, help="pink 单个子目标最大迭代次数")
    parser.add_argument("--ik-sub-step-pos", type=float, default=0.01, help="pink 子目标位置步长，米")
    parser.add_argument("--ik-sub-step-rot", type=float, default=float(np.deg2rad(6.0)), help="pink 子目标旋转步长，弧度")
    parser.add_argument("--ik-solver", default="quadprog", help="pink QP 求解器名称")
    parser.add_argument("--host", default="127.0.0.1", help="控制网页 host")
    parser.add_argument("--control-port", type=int, default=8020, help="控制网页端口")
    parser.add_argument("--meshcat-port", type=int, default=7060, help="MeshCat web 端口")
    parser.add_argument("--ghost-count", type=int, default=12, help="三维场景中显示的采样机械臂姿态数量")
    parser.add_argument("--marker-count", type=int, default=64, help="末端轨迹采样点数量")
    parser.add_argument("--root-node", default="piper_dataset_replay", help="MeshCat 根节点")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    reader = PiperDatasetReader(
        dataset_root,
        pose_column=args.pose_column,
        gripper_column=args.gripper_column or None,
    )
    parquet_file = reader.resolve_parquet_file(args.file, args.episode_index)
    episode_index = None if args.file is not None else (0 if args.episode_index is None else args.episode_index)
    config = ReplayConfig(
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        z_lift=args.z_lift,
        initial_x_offset=args.initial_x_offset,
        initial_y_offset=args.initial_y_offset,
        initial_z_offset=args.initial_z_offset,
        delta_mode=args.delta_mode,
        frequency=10.0,
        speed=1.0,
        start=args.start,
        end=args.end,
    )
    ik_options = {
        "position_cost": args.position_cost,
        "orientation_cost": args.orientation_cost,
        "posture_cost": args.posture_cost,
        "pos_tol": args.ik_pos_tol,
        "rot_tol": args.ik_rot_tol,
        "max_iters": args.ik_max_iters,
        "sub_step_pos": args.ik_sub_step_pos,
        "sub_step_rot": args.ik_sub_step_rot,
        "solver": args.ik_solver,
    }
    app = DatasetWeb3DApp(
        parquet_file=parquet_file,
        dataset_root=dataset_root,
        pose_column=args.pose_column,
        gripper_column=args.gripper_column or None,
        urdf_path=args.urdf.expanduser().resolve(),
        package_dir=args.package_dir.expanduser().resolve(),
        ee_frame=args.ee_frame,
        config=config,
        initial_joint_values=np.asarray(args.initial_joints, dtype=np.float64),
        episode_index=episode_index,
        invert_gripper=args.invert_gripper,
        ik_options=ik_options,
        meshcat_port=args.meshcat_port,
        ghost_count=args.ghost_count,
        marker_count=args.marker_count,
        root_node=args.root_node,
    )
    failed = len(app.ik_results) - app.state()["converged_count"]
    if failed:
        worst = max(app.ik_results, key=lambda result: result.position_error)
        print(f"Warning: offline IK failed on {failed}/{len(app.ik_results)} frames; worst position error {worst.position_error:.5f} m.")
    Web3DHandler.app = app
    server: ThreadingHTTPServer = make_server(args.host, args.control_port, Web3DHandler)
    host, port = server.server_address[:2]
    control_url = f"http://{host}:{port}/"
    print(f"Dataset: {app.parquet_file}")
    print(f"Mode: {args.delta_mode}")
    print(f"MeshCat: {app.meshcat_url}")
    print(f"Control panel: {control_url}")
    if not args.no_open:
        webbrowser.open(control_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
