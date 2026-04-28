#!/usr/bin/env python3
"""读取 Piper 数据集，仅在网页中显示目标末端坐标轴轨迹。"""

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
from scipy.spatial.transform import Rotation as R

from piper_ik_visualizer import html_response, json_response, make_meshcat_viewer, make_server, render_html_template
from piper_dataset_reader import (
    DEFAULT_DATASET,
    DEFAULT_POSE_COLUMN,
    ROOT,
    PiperDatasetReader,
    ReplayConfig,
    build_pose_plan,
)
from piper_ik_utils import (
    DEFAULT_INITIAL_JOINTS,
    DEFAULT_JOINTS,
    DEFAULT_URDF,
    PinocchioIK,
)


AXES = (
    ("x", np.array((1.0, 0.0, 0.0), dtype=np.float64), 0xE53935),
    ("y", np.array((0.0, 1.0, 0.0), dtype=np.float64), 0x2E7D32),
    ("z", np.array((0.0, 0.0, 1.0), dtype=np.float64), 0x1565C0),
)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("ZYX", pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def make_path_line(points: np.ndarray) -> g.Line:
    colors = np.zeros((3, points.shape[0]), dtype=np.float64)
    progress = np.linspace(0.0, 1.0, points.shape[0]) if points.shape[0] > 1 else np.array([1.0])
    colors[0, :] = 0.08 + 0.62 * progress
    colors[1, :] = 0.36 + 0.32 * (1.0 - progress)
    colors[2, :] = 0.42 + 0.22 * progress
    return g.Line(
        g.PointsGeometry(points.T, color=colors),
        g.LineBasicMaterial(vertexColors=True, linewidth=5),
    )


def make_axis_line(direction: np.ndarray, length: float, color: int, linewidth: int, opacity: float) -> g.Line:
    points = np.vstack((np.zeros(3, dtype=np.float64), direction * length)).T
    return g.Line(
        g.PointsGeometry(points),
        g.LineBasicMaterial(color=color, linewidth=linewidth, transparent=opacity < 1.0, opacity=opacity),
    )


def sample_indices(length: int, count: int) -> np.ndarray:
    if count <= 0 or length <= 0:
        return np.asarray([], dtype=int)
    return np.unique(np.linspace(0, length - 1, min(count, length)).round().astype(int))


def resolve_initial_pose(
    *,
    urdf_path: Path,
    package_dir: Path,
    ee_frame: str,
    initial_joint_values: np.ndarray,
    initial_pose: np.ndarray | None,
) -> np.ndarray:
    if initial_pose is not None:
        return np.asarray(initial_pose, dtype=np.float64).reshape(6)
    fk_model = PinocchioIK(
        urdf_path=urdf_path,
        package_dir=package_dir,
        ee_frame=ee_frame,
        active_joint_names=DEFAULT_JOINTS,
        initial_joint_values=np.asarray(initial_joint_values, dtype=np.float64),
    )
    return fk_model.forward_pose6()


class TargetAxesWeb3DApp:
    def __init__(
        self,
        *,
        parquet_file: Path,
        dataset_root: Path,
        pose_column: str,
        urdf_path: Path,
        package_dir: Path,
        ee_frame: str,
        config: ReplayConfig,
        initial_joint_values: np.ndarray,
        initial_pose: np.ndarray | None,
        episode_index: int | None,
        meshcat_port: int,
        ghost_count: int,
        marker_count: int,
        axis_length: float,
        root_node: str,
    ) -> None:
        self.lock = threading.RLock()
        self.parquet_file = parquet_file
        self.dataset_root = dataset_root
        self.pose_column = pose_column
        self.urdf_path = urdf_path
        self.package_dir = package_dir
        self.ee_frame = ee_frame
        self.config = config
        self.initial_joint_values = np.asarray(initial_joint_values, dtype=np.float64).reshape(-1)
        self.initial_pose_override = None if initial_pose is None else np.asarray(initial_pose, dtype=np.float64).reshape(6)
        self.episode_index = episode_index
        self.root_node = root_node
        self.ghost_count = ghost_count
        self.marker_count = marker_count
        self.axis_length = axis_length
        self.dataset_reader = PiperDatasetReader(dataset_root, pose_column=pose_column)
        self.fps = self.dataset_reader.fps()
        try:
            info = self.dataset_reader.load_info()
            self.total_episodes = int(info["total_episodes"]) if info.get("total_episodes") is not None else None
        except Exception:
            self.total_episodes = None
        self.deltas = np.empty((0, 6), dtype=np.float64)
        self.poses = np.empty((0, 6), dtype=np.float64)
        self.points = np.empty((0, 3), dtype=np.float64)
        self.frame = 0
        self.viewer, self.meshcat_server = make_meshcat_viewer(meshcat_port)
        self.meshcat_url = self.viewer.url()
        self.load_parquet(parquet_file, episode_index)

    def _start_pose(self) -> np.ndarray:
        return resolve_initial_pose(
            urdf_path=self.urdf_path,
            package_dir=self.package_dir,
            ee_frame=self.ee_frame,
            initial_joint_values=self.initial_joint_values,
            initial_pose=self.initial_pose_override,
        )

    def _load_axes_node(self, node: str, opacity: float, linewidth: int) -> None:
        self.viewer[f"{node}/origin"].set_object(
            g.Sphere(self.axis_length * 0.065),
            g.MeshLambertMaterial(color=0x1F1F1F, transparent=opacity < 1.0, opacity=opacity),
        )
        for axis_name, direction, color in AXES:
            self.viewer[f"{node}/{axis_name}"].set_object(
                make_axis_line(direction, self.axis_length, color, linewidth=linewidth, opacity=opacity)
            )

    def _clear_scene(self) -> None:
        self.viewer[self.root_node].delete()

    def _load_static_scene(self) -> None:
        self.viewer[f"{self.root_node}/path"].set_object(make_path_line(self.points))
        for marker_id, index in enumerate(sample_indices(len(self.points), self.marker_count)):
            self.viewer[f"{self.root_node}/markers/{marker_id:03d}"].set_object(
                g.Sphere(self.axis_length * 0.045),
                g.MeshLambertMaterial(color=0x2A7F62, transparent=True, opacity=0.55),
            )
            transform = np.eye(4)
            transform[:3, 3] = self.points[index]
            self.viewer[f"{self.root_node}/markers/{marker_id:03d}"].set_transform(transform)

        for ghost_id, index in enumerate(sample_indices(len(self.poses), self.ghost_count)):
            node = f"{self.root_node}/ghosts/{ghost_id:03d}"
            opacity = 0.16 + 0.34 * ((ghost_id + 1) / max(self.ghost_count, 1))
            self._load_axes_node(node, opacity=opacity, linewidth=2)
            self.viewer[node].set_transform(pose_to_matrix(self.poses[index]))

        self._load_axes_node(f"{self.root_node}/current_axes", opacity=1.0, linewidth=6)

    def load_parquet(self, parquet_file: Path, episode_index: int | None) -> dict[str, Any]:
        with self.lock:
            self.parquet_file = parquet_file
            self.episode_index = episode_index
            self.deltas = self.dataset_reader.load_pose_deltas(parquet_file)
            self.poses = build_pose_plan(self.deltas, self._start_pose(), self.config)
            self.points = self.poses[:, :3].copy()
            self.frame = 0
            self._clear_scene()
            self._load_static_scene()
            return self.display_frame(0)

    def load_episode(self, episode_index: int) -> dict[str, Any]:
        return self.load_parquet(self.dataset_reader.episode_parquet_path(episode_index), episode_index)

    def display_frame(self, frame: int) -> dict[str, Any]:
        with self.lock:
            frame = int(max(0, min(frame, len(self.poses) - 1)))
            self.frame = frame
            self.viewer[f"{self.root_node}/current_axes"].set_transform(pose_to_matrix(self.poses[frame]))
            return self.state()

    def state(self) -> dict[str, Any]:
        pose = self.poses[self.frame]
        return {
            "file": str(self.parquet_file),
            "episode_index": self.episode_index,
            "total_episodes": self.total_episodes,
            "meshcat_url": self.meshcat_url,
            "frame": self.frame,
            "frame_count": len(self.poses),
            "fps": self.fps,
            "delta_mode": self.config.delta_mode,
            "pos_scale": self.config.pos_scale,
            "rot_scale": self.config.rot_scale,
            "z_lift": self.config.z_lift,
            "initial_x_offset": self.config.initial_x_offset,
            "initial_y_offset": self.config.initial_y_offset,
            "initial_z_offset": self.config.initial_z_offset,
            "initial_joints": [float(value) for value in self.initial_joint_values],
            "initial_pose_override": None
            if self.initial_pose_override is None
            else [float(value) for value in self.initial_pose_override],
            "axis_length": self.axis_length,
            "pose": [float(value) for value in pose],
        }


class TargetAxesHandler(BaseHTTPRequestHandler):
    app: TargetAxesWeb3DApp

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
    return render_html_template("replay_piper_target_axes_web3d.html", meshcat_url=meshcat_url)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Display target pose axes and route without robot IK.")
    parser.add_argument("--file", "-f", type=Path, default=None, help="直接指定 episode parquet 文件；不能和 --episode-index 同时使用")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET, help="数据集根目录")
    parser.add_argument("--episode-index", type=int, default=None, help="按数据集 episode 编号选择 parquet；不填时默认 episode 0")
    parser.add_argument("--pose-column", default=DEFAULT_POSE_COLUMN, help="相对末端目标位姿列")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="仅用于从初始关节姿态做 FK 得到起始目标位姿")
    parser.add_argument("--package-dir", type=Path, default=ROOT, help="包含 piper_description 包的目录")
    parser.add_argument("--ee-frame", default="gripper_base", help="FK 起始末端 frame/link")
    parser.add_argument("--initial-joints", type=float, nargs=6, default=DEFAULT_INITIAL_JOINTS, metavar=("J1", "J2", "J3", "J4", "J5", "J6"), help="FK 起始关节姿态")
    parser.add_argument("--initial-pose", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "R", "P", "YAW"), help="直接指定起始目标姿态，设置后不读取 URDF 做 FK")
    parser.add_argument("--delta-mode", choices=("incremental", "from-start"), default="incremental", help="相对位姿解释方式")
    parser.add_argument("--start", type=int, default=0, help="开始帧")
    parser.add_argument("--end", type=int, default=None, help="结束帧")
    parser.add_argument("--pos-scale", type=float, default=0.6, help="位置增量缩放")
    parser.add_argument("--rot-scale", type=float, default=1.0, help="姿态增量缩放")
    parser.add_argument("--z-lift", type=float, default=0.0, help="起始 z 抬升")
    parser.add_argument("--initial-x-offset", type=float, default=0.0, help="起始 x 偏移，米")
    parser.add_argument("--initial-y-offset", type=float, default=0.0, help="起始 y 偏移，米")
    parser.add_argument("--initial-z-offset", type=float, default=0.0, help="起始 z 偏移，米")
    parser.add_argument("--axis-length", type=float, default=0.055, help="坐标轴长度，米")
    parser.add_argument("--host", default="127.0.0.1", help="控制网页 host")
    parser.add_argument("--control-port", type=int, default=8030, help="控制网页端口")
    parser.add_argument("--meshcat-port", type=int, default=7070, help="MeshCat web 端口")
    parser.add_argument("--ghost-count", type=int, default=24, help="轨迹中的采样坐标轴数量")
    parser.add_argument("--marker-count", type=int, default=64, help="轨迹采样点数量")
    parser.add_argument("--root-node", default="piper_target_axes", help="MeshCat 根节点")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    reader = PiperDatasetReader(dataset_root, pose_column=args.pose_column)
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
    app = TargetAxesWeb3DApp(
        parquet_file=parquet_file,
        dataset_root=dataset_root,
        pose_column=args.pose_column,
        urdf_path=args.urdf.expanduser().resolve(),
        package_dir=args.package_dir.expanduser().resolve(),
        ee_frame=args.ee_frame,
        config=config,
        initial_joint_values=np.asarray(args.initial_joints, dtype=np.float64),
        initial_pose=None if args.initial_pose is None else np.asarray(args.initial_pose, dtype=np.float64),
        episode_index=episode_index,
        meshcat_port=args.meshcat_port,
        ghost_count=args.ghost_count,
        marker_count=args.marker_count,
        axis_length=args.axis_length,
        root_node=args.root_node,
    )
    TargetAxesHandler.app = app
    server: ThreadingHTTPServer = make_server(args.host, args.control_port, TargetAxesHandler)
    host, port = server.server_address[:2]
    control_url = f"http://{host}:{port}/"
    print("Target axes only: no robot IK is run.")
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
