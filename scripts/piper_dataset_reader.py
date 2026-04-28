"""Piper 数据集读取与相对位姿轨迹构建工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset/20260420_panda_dual_pika"
DEFAULT_POSE_COLUMN = "observation.state.arm.right.end_effector_pose"
DEFAULT_GRIPPER_COLUMN = "observation.state.arm.right.end_effector_value"

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


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def convert_delta_pose_to_tool_frame(delta_pose: np.ndarray, rot_scale: float) -> tuple[np.ndarray, R]:
    delta_position_tool = DELTA_POSE_BASIS_ROTATION.apply(delta_pose[:3])
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
    base_pose = np.asarray(start_pose, dtype=np.float64).reshape(6).copy()
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
            pose = apply_delta_pose(pose, deltas[step], config.pos_scale, config.rot_scale)
        else:
            raise ValueError(f"未知 delta_mode: {config.delta_mode}")
        poses.append(pose.copy())
    return np.asarray(poses, dtype=np.float64)


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


class PiperDatasetReader:
    def __init__(
        self,
        dataset_root: Path | None = None,
        *,
        pose_column: str = DEFAULT_POSE_COLUMN,
        gripper_column: str | None = DEFAULT_GRIPPER_COLUMN,
    ) -> None:
        self.dataset_root = None if dataset_root is None else Path(dataset_root).expanduser().resolve()
        self.pose_column = pose_column
        self.gripper_column = gripper_column

    def _root(self, dataset_root: Path | None = None) -> Path:
        root = self.dataset_root if dataset_root is None else Path(dataset_root).expanduser().resolve()
        if root is None:
            raise ValueError("未设置 dataset_root")
        return root

    def list_parquet_columns(self, parquet_file: Path) -> list[str]:
        return pq.read_schema(parquet_file).names

    def column_to_numpy(self, parquet_file: Path, column_name: str) -> np.ndarray:
        parquet_file = Path(parquet_file).expanduser().resolve()
        if not parquet_file.is_file():
            raise FileNotFoundError(f"Parquet 文件不存在: {parquet_file}")
        columns = self.list_parquet_columns(parquet_file)
        if column_name not in columns:
            raise KeyError("未找到列: " + column_name + "\n可用列:\n- " + "\n- ".join(columns))
        column = pq.read_table(parquet_file, columns=[column_name]).column(column_name).combine_chunks()
        return np.asarray(column.to_pylist(), dtype=np.float64)

    def load_pose_deltas(self, parquet_file: Path, column_name: str | None = None) -> np.ndarray:
        column_name = self.pose_column if column_name is None else column_name
        deltas = self.column_to_numpy(parquet_file, column_name)
        if deltas.ndim != 2 or deltas.shape[1] != 6:
            raise ValueError(f"{column_name} 数据形状异常，期望 (N, 6)，实际为 {deltas.shape}")
        if len(deltas) == 0:
            raise ValueError(f"{column_name} 为空")
        return deltas

    def load_gripper_values(
        self,
        parquet_file: Path,
        expected_steps: int,
        column_name: str | None = None,
    ) -> np.ndarray | None:
        column_name = self.gripper_column if column_name is None else column_name
        if not column_name:
            return None
        if column_name not in self.list_parquet_columns(parquet_file):
            print(f"未找到夹爪列 {column_name}，将保持当前夹爪值。")
            return None
        values = self.column_to_numpy(parquet_file, column_name)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)
        elif values.ndim != 1:
            raise ValueError(f"{column_name} 数据形状异常，期望 (N,) 或 (N, 1)，实际为 {values.shape}")
        if len(values) != expected_steps:
            raise ValueError(f"{column_name} 长度与位姿轨迹不一致: {len(values)} != {expected_steps}")
        return np.clip(values.astype(np.float64), 0.0, 1.0)

    def load_info(self, dataset_root: Path | None = None) -> dict[str, Any]:
        info_path = self._root(dataset_root) / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"数据集 metadata 不存在: {info_path}")
        with info_path.open("r", encoding="utf-8") as fh:
            info = json.load(fh)
        if not isinstance(info, dict):
            raise ValueError(f"数据集 metadata 格式异常: {info_path}")
        return info

    def fps(self, fallback: float = 10.0, dataset_root: Path | None = None) -> float:
        try:
            info = self.load_info(dataset_root)
        except FileNotFoundError:
            return fallback
        return float(info.get("fps") or fallback)

    def episode_parquet_path(self, episode_index: int, dataset_root: Path | None = None) -> Path:
        if episode_index < 0:
            raise ValueError(f"episode_index 必须 >= 0，当前为 {episode_index}")
        root = self._root(dataset_root)
        info = self.load_info(root)
        total_episodes = info.get("total_episodes")
        if total_episodes is not None and episode_index >= int(total_episodes):
            raise ValueError(f"episode_index 超出范围: {episode_index} >= total_episodes {int(total_episodes)}")
        chunks_size = int(info.get("chunks_size") or 1000)
        episode_chunk = episode_index // chunks_size
        template = str(info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        parquet_file = root / template.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        if not parquet_file.is_file():
            raise FileNotFoundError(f"episode parquet 文件不存在: {parquet_file}")
        return parquet_file

    def resolve_parquet_file(self, file_path: Path | None, episode_index: int | None) -> Path:
        if file_path is not None and episode_index is not None:
            raise ValueError("--file 和 --episode-index 只能二选一")
        if file_path is not None:
            parquet_file = file_path.expanduser().resolve()
            if not parquet_file.is_file():
                raise FileNotFoundError(f"episode parquet 文件不存在: {parquet_file}")
            return parquet_file
        return self.episode_parquet_path(0 if episode_index is None else episode_index)

    def build_pose_plan(self, deltas: np.ndarray, start_pose: np.ndarray, config: ReplayConfig) -> np.ndarray:
        return build_pose_plan(deltas, start_pose, config)

    def load_replay_plan(
        self,
        parquet_file: Path,
        start_pose: np.ndarray,
        config: ReplayConfig,
        *,
        pose_column: str | None = None,
        gripper_column: str | None = None,
    ) -> ReplayPlan:
        deltas = self.load_pose_deltas(parquet_file, pose_column)
        gripper = self.load_gripper_values(parquet_file, len(deltas), gripper_column)
        poses = self.build_pose_plan(deltas, start_pose, config)
        return ReplayPlan(deltas=deltas, gripper=gripper, poses=poses)


def list_parquet_columns(parquet_file: Path) -> list[str]:
    return PiperDatasetReader().list_parquet_columns(parquet_file)


def load_pose_deltas(parquet_file: Path, column_name: str) -> np.ndarray:
    return PiperDatasetReader(pose_column=column_name).load_pose_deltas(parquet_file)


def load_gripper_values(parquet_file: Path, column_name: str | None, expected_steps: int) -> np.ndarray | None:
    return PiperDatasetReader(gripper_column=column_name).load_gripper_values(parquet_file, expected_steps)


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    return PiperDatasetReader(dataset_root).load_info()


def load_dataset_fps(dataset_root: Path, fallback: float = 10.0) -> float:
    return PiperDatasetReader(dataset_root).fps(fallback)


def episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return PiperDatasetReader(dataset_root).episode_parquet_path(episode_index)


def resolve_parquet_file(dataset_root: Path, file_path: Path | None, episode_index: int | None) -> Path:
    return PiperDatasetReader(dataset_root).resolve_parquet_file(file_path, episode_index)


def build_replay_plan(
    parquet_file: Path,
    pose_column: str,
    gripper_column: str | None,
    start_pose: np.ndarray,
    config: ReplayConfig,
) -> ReplayPlan:
    return PiperDatasetReader(pose_column=pose_column, gripper_column=gripper_column).load_replay_plan(
        parquet_file,
        start_pose,
        config,
    )
