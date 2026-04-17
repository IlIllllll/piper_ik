#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "20260413_panda_dual_pika"
DEFAULT_URDF = ROOT / "piper_description" / "urdf" / "piper_description.urdf"
DEFAULT_POSE_COLUMN = "observation.state.arm.right.end_effector_pose"
DEFAULT_GRIPPER_COLUMN = "observation.state.arm.right.end_effector_value"
DEFAULT_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_GRIPPER_JOINTS = ("joint7", "joint8")
DEFAULT_INITIAL_JOINTS = (0.000, 0.368, -0.692, -0.000,1.039, 0.000)
DEFAULT_TRACE_LINKS = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base")

# Dataset delta poses are recorded in a frame that differs from the tool frame by a -pi/2 rotation
# around Y. Translation and rotation deltas both need this similarity transform to be re-expressed
# in the current end-effector (tool) frame before integration.
DELTA_POSE_BASIS_ROTATION = R.from_euler("Y", -np.pi / 2.0, degrees=False)


@dataclass
class ReplayConfig:
    pos_scale: float
    rot_scale: float
    z_lift: float
    initial_x_offset: float
    initial_y_offset: float
    initial_z_offset: float
    delta_mode: str
    frequency: float
    speed: float
    start: int
    end: int | None


@dataclass
class ReplayPlan:
    deltas: np.ndarray
    gripper: np.ndarray | None
    poses: np.ndarray


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error: float
    rotation_error: float


def list_parquet_columns(parquet_file: Path) -> list[str]:
    return pq.read_schema(parquet_file).names


def _column_to_numpy(parquet_file: Path, column_name: str) -> np.ndarray:
    if not parquet_file.is_file():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_file}")
    columns = list_parquet_columns(parquet_file)
    if column_name not in columns:
        raise KeyError("未找到列: " + column_name + "\n可用列:\n- " + "\n- ".join(columns))
    column = pq.read_table(parquet_file, columns=[column_name]).column(column_name).combine_chunks()
    return np.asarray(column.to_pylist(), dtype=np.float64)


def load_pose_deltas(parquet_file: Path, column_name: str) -> np.ndarray:
    deltas = _column_to_numpy(parquet_file, column_name)
    if deltas.ndim != 2 or deltas.shape[1] != 6:
        raise ValueError(f"{column_name} 数据形状异常，期望 (N, 6)，实际为 {deltas.shape}")
    if len(deltas) == 0:
        raise ValueError(f"{column_name} 为空")
    return deltas


def load_gripper_values(parquet_file: Path, column_name: str | None, expected_steps: int) -> np.ndarray | None:
    if not column_name:
        return None
    if column_name not in list_parquet_columns(parquet_file):
        print(f"未找到夹爪列 {column_name}，将保持当前夹爪值。")
        return None
    values = _column_to_numpy(parquet_file, column_name)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.reshape(-1)
    elif values.ndim != 1:
        raise ValueError(f"{column_name} 数据形状异常，期望 (N,) 或 (N, 1)，实际为 {values.shape}")
    if len(values) != expected_steps:
        raise ValueError(f"{column_name} 长度与位姿轨迹不一致: {len(values)} != {expected_steps}")
    return np.clip(values.astype(np.float64), 0.0, 1.0)


def load_dataset_fps(dataset_root: Path, fallback: float = 10.0) -> float:
    try:
        info = load_dataset_info(dataset_root)
    except FileNotFoundError:
        return fallback
    return float(info.get("fps") or fallback)


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"数据集 metadata 不存在: {info_path}")
    with info_path.open("r", encoding="utf-8") as fh:
        info = json.load(fh)
    if not isinstance(info, dict):
        raise ValueError(f"数据集 metadata 格式异常: {info_path}")
    return info


def episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    if episode_index < 0:
        raise ValueError(f"episode_index 必须 >= 0，当前为 {episode_index}")
    info = load_dataset_info(dataset_root)
    total_episodes = info.get("total_episodes")
    if total_episodes is not None and episode_index >= int(total_episodes):
        raise ValueError(f"episode_index 超出范围: {episode_index} >= total_episodes {int(total_episodes)}")
    chunks_size = int(info.get("chunks_size") or 1000)
    episode_chunk = episode_index // chunks_size
    template = str(info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    parquet_file = dataset_root / template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    if not parquet_file.is_file():
        raise FileNotFoundError(f"episode parquet 文件不存在: {parquet_file}")
    return parquet_file


def resolve_parquet_file(dataset_root: Path, file_path: Path | None, episode_index: int | None) -> Path:
    if file_path is not None and episode_index is not None:
        raise ValueError("--file 和 --episode-index 只能二选一")
    if file_path is not None:
        parquet_file = file_path.expanduser().resolve()
        if not parquet_file.is_file():
            raise FileNotFoundError(f"episode parquet 文件不存在: {parquet_file}")
        return parquet_file
    return episode_parquet_path(dataset_root, 0 if episode_index is None else episode_index)


def pose6_to_se3(pose: np.ndarray) -> pin.SE3:
    rotation = R.from_euler("ZYX", pose[3:]).as_matrix()
    return pin.SE3(rotation, pose[:3].copy())


def se3_to_pose6(transform: pin.SE3) -> np.ndarray:
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = transform.translation
    pose[3:] = R.from_matrix(transform.rotation).as_euler("ZYX")
    return pose


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def convert_delta_pose_to_tool_frame(delta_pose: np.ndarray, rot_scale: float) -> tuple[np.ndarray, R]:
    # Translation delta expressed in the recorded basis -> rotated into the tool frame.
    delta_position_tool = DELTA_POSE_BASIS_ROTATION.apply(delta_pose[:3])
    # Rotation delta expressed in the recorded basis -> similarity transform into the tool frame.
    delta_rotation_in_recorded_basis = R.from_euler("ZYX", delta_pose[3:] * rot_scale, degrees=False)
    delta_rotation = (
        DELTA_POSE_BASIS_ROTATION
        * delta_rotation_in_recorded_basis
        * DELTA_POSE_BASIS_ROTATION.inv()
    )
    return delta_position_tool, delta_rotation


def apply_delta_pose(current_pose: np.ndarray, delta_pose: np.ndarray, pos_scale: float, rot_scale: float) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float64).reshape(6)
    delta_pose = np.asarray(delta_pose, dtype=np.float64).reshape(6)

    current_rot = R.from_euler("ZYX", current_pose[3:], degrees=False)
    delta_position_tool, delta_rot = convert_delta_pose_to_tool_frame(delta_pose, rot_scale)

    target = current_pose.copy()
    target[:3] += current_rot.apply(delta_position_tool * pos_scale)
    target[3:] = wrap_to_pi((current_rot * delta_rot).as_euler("ZYX", degrees=False))
    return target


def slice_range(total_steps: int, start: int, end: int | None) -> tuple[int, int]:
    start = max(0, start)
    end = total_steps if end is None else min(total_steps, end)
    if start >= end:
        raise ValueError(f"无效的回放范围: start={start}, end={end}, total={total_steps}")
    return start, end


def build_pose_plan(deltas: np.ndarray, start_pose: np.ndarray, config: ReplayConfig) -> np.ndarray:
    start, end = slice_range(len(deltas), config.start, config.end)
    base_pose = start_pose.copy()
    base_pose[:3] += np.array(
        (config.initial_x_offset, config.initial_y_offset, config.initial_z_offset),
        dtype=np.float64,
    )
    base_pose[2] += config.z_lift
    pose = base_pose.copy()
    poses = [base_pose.copy()]
    for step in range(start, end):
        if config.delta_mode == "from-start":
            pose = apply_delta_pose(base_pose, deltas[step], config.pos_scale, config.rot_scale)
        elif config.delta_mode == "incremental":
            # Dataset rows are per-frame relative motions by default.
            pose = apply_delta_pose(pose, deltas[step], config.pos_scale, config.rot_scale)
        else:
            raise ValueError(f"未知 delta_mode: {config.delta_mode}")
        poses.append(pose.copy())
    return np.asarray(poses, dtype=np.float64)


def build_replay_plan(
    parquet_file: Path,
    pose_column: str,
    gripper_column: str | None,
    start_pose: np.ndarray,
    config: ReplayConfig,
) -> ReplayPlan:
    deltas = load_pose_deltas(parquet_file, pose_column)
    gripper = load_gripper_values(parquet_file, gripper_column, len(deltas))
    poses = build_pose_plan(deltas, start_pose, config)
    return ReplayPlan(deltas=deltas, gripper=gripper, poses=poses)


def active_velocity_mask(model: pin.Model, active_joint_names: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(model.nv, dtype=bool)
    for name in active_joint_names:
        joint_id = model.getJointId(name)
        if joint_id >= len(model.names) or model.names[joint_id] != name:
            raise ValueError(f"URDF 中未找到关节 {name!r}")
        joint = model.joints[joint_id]
        mask[joint.idx_v : joint.idx_v + joint.nv] = True
    return mask


def joint_vector_from_names(model: pin.Model, joint_names: tuple[str, ...], values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != len(joint_names):
        raise ValueError(f"初始关节数量不匹配: {len(values)} != {len(joint_names)}")
    q = pin.neutral(model)
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    for index in range(model.nq):
        if np.isfinite(lower[index]) and np.isfinite(upper[index]) and upper[index] > lower[index]:
            q[index] = 0.5 * (lower[index] + upper[index])
    for joint_name, value in zip(joint_names, values):
        joint_id = model.getJointId(joint_name)
        joint = model.joints[joint_id]
        q[joint.idx_q] = value
    return clip_to_limits(model, q)


def gripper_sequence_for_poses(
    gripper_values: np.ndarray | None,
    total_steps: int,
    start: int,
    end: int | None,
) -> np.ndarray | None:
    if gripper_values is None:
        return None
    start, end = slice_range(total_steps, start, end)
    values = np.asarray(gripper_values, dtype=np.float64).reshape(-1)[start:end]
    if len(values) == 0:
        return None
    return np.concatenate((values[:1], values))


def gripper_joint_position(lower: float, upper: float, opening: float) -> float:
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return float(opening)
    opening = float(np.clip(opening, 0.0, 1.0))
    if lower <= 0.0 <= upper:
        closed = 0.0
        opened = upper if abs(upper) >= abs(lower) else lower
    elif abs(lower) <= abs(upper):
        closed = lower
        opened = upper
    else:
        closed = upper
        opened = lower
    return closed + opening * (opened - closed)


def apply_gripper_values_to_qs(
    model: pin.Model,
    qs: np.ndarray,
    gripper_values: np.ndarray | None,
    gripper_joint_names: tuple[str, ...] = DEFAULT_GRIPPER_JOINTS,
    invert: bool = False,
) -> np.ndarray:
    if gripper_values is None:
        return qs
    values = np.asarray(gripper_values, dtype=np.float64).reshape(-1)
    if len(values) != len(qs):
        raise ValueError(f"夹爪值数量与姿态数量不一致: {len(values)} != {len(qs)}")
    if invert:
        values = 1.0 - values
    result = qs.copy()
    for joint_name in gripper_joint_names:
        joint_id = model.getJointId(joint_name)
        if joint_id >= len(model.names) or model.names[joint_id] != joint_name:
            raise ValueError(f"URDF 中未找到夹爪关节 {joint_name!r}")
        joint = model.joints[joint_id]
        if joint.nq != 1:
            raise ValueError(f"夹爪关节 {joint_name!r} 不是单自由度关节")
        idx_q = joint.idx_q
        lower = model.lowerPositionLimit[idx_q]
        upper = model.upperPositionLimit[idx_q]
        for frame_index, opening in enumerate(values):
            result[frame_index, idx_q] = gripper_joint_position(lower, upper, opening)
    return clip_to_limits(model, result)


def home_configuration(model: pin.Model) -> np.ndarray:
    q = pin.neutral(model)
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    for index in range(model.nq):
        if np.isfinite(lower[index]) and np.isfinite(upper[index]) and upper[index] > lower[index]:
            q[index] = 0.5 * (lower[index] + upper[index])
    return clip_to_limits(model, q)


def clip_to_limits(model: pin.Model, q: np.ndarray) -> np.ndarray:
    clipped = q.copy()
    for index in range(model.nq):
        lower = model.lowerPositionLimit[index]
        upper = model.upperPositionLimit[index]
        if np.isfinite(lower):
            clipped[..., index] = np.maximum(clipped[..., index], lower)
        if np.isfinite(upper):
            clipped[..., index] = np.minimum(clipped[..., index], upper)
    return clipped


class PinocchioIK:
    def __init__(
        self,
        urdf_path: Path,
        package_dir: Path,
        ee_frame: str,
        active_joint_names: tuple[str, ...],
        q0: np.ndarray | None = None,
        initial_joint_values: np.ndarray | None = None,
    ) -> None:
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            str(urdf_path),
            str(package_dir),
        )
        self.data = self.model.createData()
        self.ee_frame = ee_frame
        self.ee_frame_id = self.model.getFrameId(ee_frame)
        if self.ee_frame_id >= len(self.model.frames) or self.model.frames[self.ee_frame_id].name != ee_frame:
            raise ValueError(f"URDF 中未找到末端 frame/link: {ee_frame}")
        self.active_mask = active_velocity_mask(self.model, active_joint_names)
        if q0 is not None:
            self.q = clip_to_limits(self.model, q0)
        elif initial_joint_values is not None:
            self.q = joint_vector_from_names(self.model, active_joint_names, initial_joint_values)
        else:
            self.q = home_configuration(self.model)

    def forward_pose6(self, q: np.ndarray | None = None) -> np.ndarray:
        if q is not None:
            self.q = q.copy()
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        return se3_to_pose6(self.data.oMf[self.ee_frame_id])

    def solve(
        self,
        target_pose: np.ndarray,
        max_iterations: int = 250,
        tolerance: float = 1e-4,
        damping: float = 1e-4,
        dt: float = 0.45,
        max_velocity: float = 0.35,
        position_weight: float = 1.0,
        rotation_weight: float = 0.55,
    ) -> IKResult:
        target = pose6_to_se3(target_pose)
        weights = np.array(
            [position_weight, position_weight, position_weight, rotation_weight, rotation_weight, rotation_weight],
            dtype=np.float64,
        )
        identity6 = np.eye(6)
        result = IKResult(self.q.copy(), False, 0, math.inf, math.inf)
        for iteration in range(1, max_iterations + 1):
            pin.forwardKinematics(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.ee_frame_id]
            error = pin.log6(current.actInv(target)).vector
            weighted_error = weights * error
            if np.linalg.norm(weighted_error) < tolerance:
                result = IKResult(self.q.copy(), True, iteration - 1, float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])))
                break
            jacobian = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_frame_id, pin.LOCAL)
            jacobian[:, ~self.active_mask] = 0.0
            weighted_jacobian = weights[:, None] * jacobian
            lhs = weighted_jacobian @ weighted_jacobian.T + damping * identity6
            velocity = weighted_jacobian.T @ np.linalg.solve(lhs, weighted_error)
            velocity[~self.active_mask] = 0.0
            velocity_norm = np.linalg.norm(velocity)
            if velocity_norm > max_velocity:
                velocity *= max_velocity / velocity_norm
            self.q = clip_to_limits(self.model, pin.integrate(self.model, self.q, velocity * dt))
            result = IKResult(self.q.copy(), False, iteration, float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])))
        return result

    def trace_points(self, q: np.ndarray, link_names: tuple[str, ...]) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        points = []
        for name in link_names:
            frame_id = self.model.getFrameId(name)
            if frame_id < len(self.model.frames) and self.model.frames[frame_id].name == name:
                points.append(self.data.oMf[frame_id].translation.copy())
        return np.asarray(points, dtype=np.float64)


def solve_ik_sequence(ik: PinocchioIK, poses: np.ndarray) -> tuple[np.ndarray, list[IKResult]]:
    qs = []
    results = []
    for pose in poses:
        results.append(ik.solve(pose))
        qs.append(ik.q.copy())
    return np.asarray(qs), results


def project_points(points: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray, width: int, height: int, padding: int) -> list[tuple[int, int]]:
    # Oblique projection gives a readable arm silhouette without a 3D renderer.
    projected = np.column_stack((points[:, 0] - 0.34 * points[:, 1], points[:, 2] + 0.18 * points[:, 1]))
    all_min = np.array((bounds_min[0] - 0.34 * bounds_max[1], bounds_min[2] + 0.18 * bounds_min[1]))
    all_max = np.array((bounds_max[0] - 0.34 * bounds_min[1], bounds_max[2] + 0.18 * bounds_max[1]))
    span = np.maximum(all_max - all_min, 1e-6)
    scale = min((width - 2 * padding) / span[0], (height - 2 * padding) / span[1])
    center = 0.5 * (all_min + all_max)
    canvas_center = np.array((width / 2.0, height / 2.0))
    pixels = []
    for point in projected:
        xy = (point - center) * scale
        pixel = canvas_center + np.array((xy[0], -xy[1]))
        pixels.append((int(round(pixel[0])), int(round(pixel[1]))))
    return pixels


def draw_scene(
    chains: list[np.ndarray],
    ee_points: np.ndarray,
    frame_index: int,
    total_frames: int,
    width: int,
    height: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    title: str,
    accumulate: bool,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (244, 247, 245))
    draw = ImageDraw.Draw(image, "RGBA")
    padding = 74
    max_chain = frame_index + 1 if accumulate else len(chains)

    grid_color = (202, 212, 205, 110)
    for gx in range(padding, width - padding + 1, 80):
        draw.line([(gx, padding), (gx, height - padding)], fill=grid_color, width=1)
    for gy in range(padding, height - padding + 1, 80):
        draw.line([(padding, gy), (width - padding, gy)], fill=grid_color, width=1)

    trail_count = max(2, frame_index + 1)
    trail_pixels = project_points(ee_points[:trail_count], bounds_min, bounds_max, width, height, padding)
    if len(trail_pixels) >= 2:
        draw.line(trail_pixels, fill=(27, 116, 83, 210), width=4, joint="curve")

    for idx in range(max_chain):
        if accumulate:
            alpha = int(28 + 170 * ((idx + 1) / max_chain) ** 1.7)
            width_line = 2 if idx < max_chain - 1 else 5
            color = (39, 97, 150, alpha)
        else:
            alpha = int(32 + 156 * ((idx + 1) / max_chain))
            width_line = 2
            color = (39, 97, 150, alpha)
        pixels = project_points(chains[idx], bounds_min, bounds_max, width, height, padding)
        if len(pixels) >= 2:
            draw.line(pixels, fill=color, width=width_line, joint="curve")
        for px in pixels:
            radius = max(2, width_line)
            draw.ellipse((px[0] - radius, px[1] - radius, px[0] + radius, px[1] + radius), fill=(22, 33, 28, min(255, alpha + 35)))

    current_ee = trail_pixels[-1]
    draw.ellipse((current_ee[0] - 6, current_ee[1] - 6, current_ee[0] + 6, current_ee[1] + 6), fill=(28, 137, 87, 245))

    font = ImageFont.load_default()
    draw.text((22, 20), title, fill=(25, 35, 30, 255), font=font)
    draw.text((22, 42), f"frame {frame_index + 1}/{total_frames}", fill=(75, 88, 80, 255), font=font)
    return image


def make_timelapse(
    ik: PinocchioIK,
    qs: np.ndarray,
    output_dir: Path,
    prefix: str,
    sample_count: int,
    width: int,
    height: int,
    fps: float,
    make_video: bool,
    link_names: tuple[str, ...] = DEFAULT_TRACE_LINKS,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(qs) == 0:
        raise ValueError("没有可用于生成影像的关节轨迹")
    indices = np.linspace(0, len(qs) - 1, min(sample_count, len(qs))).round().astype(int)
    chains = [ik.trace_points(qs[index], link_names) for index in indices]
    ee_points = np.asarray([chain[-1] for chain in chains], dtype=np.float64)
    all_points = np.concatenate(chains, axis=0)
    margin = np.array((0.08, 0.08, 0.08))
    bounds_min = np.min(all_points, axis=0) - margin
    bounds_max = np.max(all_points, axis=0) + margin
    title = f"Piper dataset replay timelapse - {prefix}"
    still = draw_scene(
        chains=chains,
        ee_points=ee_points,
        frame_index=len(chains) - 1,
        total_frames=len(chains),
        width=width,
        height=height,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        title=title,
        accumulate=False,
    )
    png_path = output_dir / f"{prefix}_timelapse.png"
    still.save(png_path)

    video_path = None
    if make_video:
        frames = [
            np.asarray(
                draw_scene(
                    chains=chains,
                    ee_points=ee_points,
                    frame_index=index,
                    total_frames=len(chains),
                    width=width,
                    height=height,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    title=title,
                    accumulate=True,
                )
            )
            for index in range(len(chains))
        ]
        video_path = output_dir / f"{prefix}_timelapse.mp4"
        iio.imwrite(video_path, frames, fps=fps, codec="libx264", macro_block_size=8)
    return png_path, video_path


def import_robot_factory(robot_project_root: Path | None):
    candidates = []
    if robot_project_root is not None:
        candidates.append(robot_project_root)
    env_root = os.environ.get("PIPER_ROBOT_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([ROOT, ROOT.parent, ROOT.parent / "lcp" / "pika_setup", ROOT.parent / ".remote_edit"])
    for candidate in candidates:
        if (candidate / "app").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        importlib.import_module("app.devices")
        from app.utils.abstract_fractory import RobotFactory
    except Exception as exc:
        raise RuntimeError(
            "无法导入真实机械臂控制接口 app.devices/RobotFactory。"
            "请用 --robot-project-root 指向包含 app/ 的控制工程，或设置 PIPER_ROBOT_PROJECT_ROOT。"
        ) from exc
    return RobotFactory


def build_arm(ip: str, robot_project_root: Path | None):
    robot_factory = import_robot_factory(robot_project_root)
    config = robot_factory.getConfigFactory("arm").create("piper_arm", name="piper_right_replay", dof=6, ip=ip)
    return robot_factory.getDeviceFactory("arm").create("piper_arm", config)


def execute_on_arm(
    arm: Any,
    deltas: np.ndarray,
    gripper_values: np.ndarray | None,
    config: ReplayConfig,
    max_consecutive_failures: int,
) -> None:
    if not arm.connect(timeout=10.0):
        raise RuntimeError("Piper 连接失败")
    if not arm.enable():
        raise RuntimeError("Piper 使能失败")
    state = arm.get_state()
    if state.joint_positions is None:
        raise RuntimeError("无法读取 Piper 当前关节角")
    current_joints = np.asarray(state.joint_positions, dtype=np.float64)
    current_gripper = (
        np.asarray(state.end_effector_value, dtype=np.float64)
        if state.end_effector_value is not None
        else np.zeros(1, dtype=np.float64)
    )
    current_pose = np.asarray(arm.get_forward_kinematics(current_joints), dtype=np.float64)
    if current_pose.shape != (6,) or np.allclose(current_pose, 0.0):
        raise RuntimeError("Piper FK 不可用，无法从当前状态开始做相对位姿回放")

    if config.z_lift:
        lifted = current_pose.copy()
        lifted[2] += config.z_lift
        lifted_joints = arm.get_inverse_kinematics(lifted, current_joints=current_joints)
        if lifted_joints is not None and arm.move_joints(lifted_joints, current_gripper, follow=False):
            current_pose = lifted
            current_joints = np.asarray(lifted_joints, dtype=np.float64)
            time.sleep(1.5)
        else:
            print("预抬升 IK/发送失败，继续直接回放。")
    offset = np.array(
        (config.initial_x_offset, config.initial_y_offset, config.initial_z_offset),
        dtype=np.float64,
    )
    if not np.allclose(offset, 0.0):
        shifted = current_pose.copy()
        shifted[:3] += offset
        shifted_joints = arm.get_inverse_kinematics(shifted, current_joints=current_joints)
        if shifted_joints is not None and arm.move_joints(shifted_joints, current_gripper, follow=False):
            current_pose = shifted
            current_joints = np.asarray(shifted_joints, dtype=np.float64)
            time.sleep(1.0)
        else:
            print(f"初始偏移 IK/发送失败，继续使用当前位姿。offset={offset}")
    base_pose = current_pose.copy()

    start, end = slice_range(len(deltas), config.start, config.end)
    interval = 1.0 / (config.frequency * config.speed)
    started_at = time.monotonic()
    consecutive_failures = 0
    for step in range(start, end):
        if config.delta_mode == "from-start":
            target_pose = apply_delta_pose(base_pose, deltas[step], config.pos_scale, config.rot_scale)
        elif config.delta_mode == "incremental":
            target_pose = apply_delta_pose(current_pose, deltas[step], config.pos_scale, config.rot_scale)
        else:
            raise ValueError(f"未知 delta_mode: {config.delta_mode}")
        gripper = (
            current_gripper
            if gripper_values is None
            else np.asarray([float(np.clip(gripper_values[step], 0.0, 1.0))], dtype=np.float64)
        )
        joints = arm.get_inverse_kinematics(target_pose, current_joints=current_joints)
        if joints is None or not arm.move_joints(joints, gripper, follow=True):
            consecutive_failures += 1
            print(f"第 {step} 步执行失败，连续失败 {consecutive_failures}/{max_consecutive_failures}")
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError("连续 IK/发送失败次数过多，停止回放")
        else:
            consecutive_failures = 0
            current_pose = target_pose
            current_joints = np.asarray(joints, dtype=np.float64)
            current_gripper = gripper
        if (step - start + 1) % 20 == 0 or step == end - 1:
            print(f"执行进度: {step - start + 1}/{end - start}")
        remaining = started_at + (step - start + 1) * interval - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取 Piper 数据集，复现相对位姿轨迹，并生成延时摄影式轨迹影像。")
    parser.add_argument("--file", "-f", type=Path, default=None, help="直接指定 episode parquet 文件；不能和 --episode-index 同时使用")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET, help="数据集根目录，用于读取 fps metadata")
    parser.add_argument("--episode-index", type=int, default=None, help="按数据集 episode 编号选择 parquet；不填时默认 episode 0")
    parser.add_argument("--pose-column", default=DEFAULT_POSE_COLUMN, help="相对末端位姿列")
    parser.add_argument("--gripper-column", default=DEFAULT_GRIPPER_COLUMN, help="夹爪列，留空则保持当前夹爪值")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="用于离线 IK 和轨迹影像的 URDF")
    parser.add_argument("--package-dir", type=Path, default=ROOT, help="包含 piper_description 包的目录")
    parser.add_argument("--ee-frame", default="gripper_base", help="IK 末端 frame/link")
    parser.add_argument("--ip", default="can_right", help="真实 Piper CAN 接口")
    parser.add_argument("--robot-project-root", type=Path, default=None, help="包含 app/ 的真实机械臂控制工程根目录")
    parser.add_argument("--execute", action="store_true", help="真正连接并驱动机械臂；默认只做离线 IK 和影像生成")
    parser.add_argument("--start", type=int, default=0, help="开始帧")
    parser.add_argument("--end", type=int, default=None, help="结束帧，不填则到 episode 末尾")
    parser.add_argument("--frequency", type=float, default=10.0, help="真实机械臂发送频率 Hz")
    parser.add_argument("--speed", type=float, default=1.0, help="真实机械臂回放速度倍率")
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
    parser.add_argument(
        "--delta-mode",
        choices=("from-start", "incremental"),
        default="incremental",
        help="incremental: 每帧相对上一帧并逐帧累加；from-start: 每帧相对 episode 起点",
    )
    parser.add_argument("--z-lift", type=float, default=0.0, help="回放前 z 轴抬升高度，米")
    parser.add_argument("--initial-x-offset", type=float, default=0.0, help="初始位姿 x 偏移，米；向后 20cm 可设为 -0.2")
    parser.add_argument("--initial-y-offset", type=float, default=0.0, help="初始位姿 y 偏移，米")
    parser.add_argument("--initial-z-offset", type=float, default=0.0, help="初始位姿 z 偏移，米")
    parser.add_argument("--max-consecutive-failures", type=int, default=5, help="真实执行连续失败阈值")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="轨迹影像输出目录")
    parser.add_argument("--sample-count", type=int, default=48, help="延时摄影中抽取的姿态数量")
    parser.add_argument("--width", type=int, default=1280, help="输出宽度")
    parser.add_argument("--height", type=int, default=824, help="输出高度")
    parser.add_argument("--video", action="store_true", help="同时输出累积轨迹 mp4")
    parser.add_argument("--video-fps", type=float, default=None, help="mp4 fps；默认使用数据集 fps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    parquet_file = resolve_parquet_file(dataset_root, args.file, args.episode_index)
    fps = load_dataset_fps(dataset_root)
    config = ReplayConfig(
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        z_lift=args.z_lift,
        initial_x_offset=args.initial_x_offset,
        initial_y_offset=args.initial_y_offset,
        initial_z_offset=args.initial_z_offset,
        delta_mode=args.delta_mode,
        frequency=args.frequency,
        speed=args.speed,
        start=args.start,
        end=args.end,
    )
    print("读取数据集:", parquet_file)
    deltas = load_pose_deltas(parquet_file, args.pose_column)
    gripper_values = load_gripper_values(parquet_file, args.gripper_column or None, len(deltas))
    print(f"位姿帧数: {len(deltas)}, 数据集 fps: {fps:.2f}")
    print(f"相对位姿模式: {args.delta_mode}")
    print(f"位置缩放: {args.pos_scale:.3f}, 姿态缩放: {args.rot_scale:.3f}, z 抬升: {args.z_lift:.3f} m")
    print(
        "初始位姿偏移: "
        f"x={args.initial_x_offset:.3f}, y={args.initial_y_offset:.3f}, z={args.initial_z_offset:.3f} m"
    )
    initial_joints = np.asarray(args.initial_joints, dtype=np.float64)
    print(f"离线初始关节: {np.array2string(initial_joints, precision=3)}")
    print(f"首帧 delta: {np.array2string(deltas[0], precision=5)}")

    ik = PinocchioIK(
        urdf_path=args.urdf,
        package_dir=args.package_dir,
        ee_frame=args.ee_frame,
        active_joint_names=DEFAULT_JOINTS,
        initial_joint_values=initial_joints,
    )
    offline_start_pose = ik.forward_pose6()
    plan = ReplayPlan(
        deltas=deltas,
        gripper=gripper_values,
        poses=build_pose_plan(deltas, offline_start_pose, config),
    )
    qs, ik_results = solve_ik_sequence(ik, plan.poses)
    failed = [index for index, result in enumerate(ik_results) if not result.converged]
    print(f"离线 IK: {len(ik_results) - len(failed)}/{len(ik_results)} 收敛")
    if failed:
        worst = max(ik_results, key=lambda item: item.position_error)
        print(f"未收敛帧数: {len(failed)}，最大位置误差约 {worst.position_error:.5f} m")

    prefix = parquet_file.stem
    png_path, video_path = make_timelapse(
        ik=ik,
        qs=qs,
        output_dir=args.output_dir,
        prefix=prefix,
        sample_count=args.sample_count,
        width=args.width,
        height=args.height,
        fps=args.video_fps or fps,
        make_video=args.video,
    )
    print(f"轨迹延时影像 PNG: {png_path}")
    if video_path:
        print(f"轨迹累积视频 MP4: {video_path}")

    if args.execute:
        print("开始真实机械臂回放。")
        arm = build_arm(args.ip, args.robot_project_root)
        try:
            execute_on_arm(
                arm=arm,
                deltas=deltas,
                gripper_values=gripper_values,
                config=config,
                max_consecutive_failures=args.max_consecutive_failures,
            )
        finally:
            try:
                if getattr(arm, "is_enabled", False):
                    arm.disable()
            except Exception:
                pass
            try:
                if getattr(arm, "is_connected", False):
                    arm.disconnect()
            except Exception:
                pass
    else:
        print("未加 --execute，本次未连接真实机械臂。")
    return 0


if __name__ == "__main__":
    def _signal_to_interrupt(*_args):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _signal_to_interrupt)
    raise SystemExit(main())
