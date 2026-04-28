"""Piper 机械臂 Pinocchio/Pink 逆解与关节工具。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R, Slerp

try:
    import pink
    from pink.tasks import FrameTask, PostureTask
except ImportError:
    print("未安装 pink 依赖，请先执行 `pip install -r requirements-ik.txt`。")
    pink = None
    FrameTask = None
    PostureTask = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "piper_description" / "urdf" / "piper_description.urdf"
DEFAULT_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
DEFAULT_GRIPPER_JOINTS = ("joint7", "joint8")
DEFAULT_INITIAL_JOINTS = (0.000, 0.368, -0.692, -0.000, 1.039, 0.000)


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error: float
    rotation_error: float
    weighted_error: float = math.inf


def _package_dirs_arg(package_dir: Path | str | Sequence[Path | str]) -> str | list[str]:
    if isinstance(package_dir, (str, Path)):
        return str(Path(package_dir).expanduser().resolve())
    paths = [str(Path(path).expanduser().resolve()) for path in package_dir]
    if not paths:
        raise ValueError("package_dir 不能为空")
    return paths[0] if len(paths) == 1 else paths


def pose6_to_se3(pose: np.ndarray) -> pin.SE3:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    rotation = R.from_euler("ZYX", pose[3:]).as_matrix()
    return pin.SE3(rotation, pose[:3].copy())


def se3_to_pose6(transform: pin.SE3) -> np.ndarray:
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = transform.translation
    pose[3:] = R.from_matrix(transform.rotation).as_euler("ZYX")
    return pose


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def interpolate_pose6(start_pose: np.ndarray, target_pose: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    start_pose = np.asarray(start_pose, dtype=np.float64).reshape(6)
    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(6)
    interpolated = np.zeros(6, dtype=np.float64)
    interpolated[:3] = (1.0 - alpha) * start_pose[:3] + alpha * target_pose[:3]
    rotations = R.from_quat(
        np.vstack(
            (
                R.from_euler("ZYX", start_pose[3:], degrees=False).as_quat(),
                R.from_euler("ZYX", target_pose[3:], degrees=False).as_quat(),
            )
        )
    )
    slerp = Slerp([0.0, 1.0], rotations)
    interpolated[3:] = wrap_to_pi(slerp([alpha])[0].as_euler("ZYX", degrees=False))
    return interpolated


def frame_id(model: pin.Model, name: str) -> int:
    result = model.getFrameId(name)
    if result >= len(model.frames) or model.frames[result].name != name:
        frame_names = ", ".join(frame.name for frame in model.frames[:80])
        raise ValueError(f"URDF 中未找到 frame/link {name!r}。可用 frame 前 80 个: {frame_names}")
    return result


def joint_id(model: pin.Model, name: str) -> int:
    result = model.getJointId(name)
    if result >= len(model.joints) or model.names[result] != name:
        joint_names = ", ".join(model.names)
        raise ValueError(f"URDF 中未找到关节 {name!r}。可用关节: {joint_names}")
    return result


def active_velocity_mask(model: pin.Model, active_joint_names: tuple[str, ...]) -> np.ndarray:
    mask = np.zeros(model.nv, dtype=bool)
    if not active_joint_names:
        mask[:] = True
        return mask
    for name in active_joint_names:
        joint = model.joints[joint_id(model, name)]
        mask[joint.idx_v : joint.idx_v + joint.nv] = True
    return mask


def home_configuration(model: pin.Model) -> np.ndarray:
    q = pin.neutral(model)
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit
    for index in range(model.nq):
        if np.isfinite(lower[index]) and np.isfinite(upper[index]) and upper[index] > lower[index]:
            q[index] = 0.5 * (lower[index] + upper[index])
    return clip_to_limits(model, q)


def clip_to_limits(model: pin.Model, q: np.ndarray) -> np.ndarray:
    clipped = np.asarray(q, dtype=np.float64).copy()
    for index in range(model.nq):
        lower = model.lowerPositionLimit[index]
        upper = model.upperPositionLimit[index]
        if np.isfinite(lower):
            clipped[..., index] = np.maximum(clipped[..., index], lower)
        if np.isfinite(upper):
            clipped[..., index] = np.minimum(clipped[..., index], upper)
    return clipped


def joint_vector_from_names(model: pin.Model, joint_names: tuple[str, ...], values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != len(joint_names):
        raise ValueError(f"初始关节数量不匹配: {len(values)} != {len(joint_names)}")
    q = home_configuration(model)
    for joint_name, value in zip(joint_names, values):
        joint = model.joints[joint_id(model, joint_name)]
        if joint.nq != 1:
            raise ValueError(f"关节 {joint_name!r} 不是单自由度关节")
        q[joint.idx_q] = value
    return clip_to_limits(model, q)


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
    result = np.asarray(qs, dtype=np.float64).copy()
    for joint_name in gripper_joint_names:
        joint = model.joints[joint_id(model, joint_name)]
        if joint.nq != 1:
            raise ValueError(f"夹爪关节 {joint_name!r} 不是单自由度关节")
        idx_q = joint.idx_q
        lower = model.lowerPositionLimit[idx_q]
        upper = model.upperPositionLimit[idx_q]
        for frame_index, opening in enumerate(values):
            result[frame_index, idx_q] = gripper_joint_position(lower, upper, opening)
    return clip_to_limits(model, result)


class PinocchioIK:
    def __init__(
        self,
        urdf_path: Path,
        package_dir: Path | str | Sequence[Path | str],
        ee_frame: str,
        active_joint_names: tuple[str, ...] = DEFAULT_JOINTS,
        q0: np.ndarray | None = None,
        initial_joint_values: np.ndarray | None = None,
    ) -> None:
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            str(Path(urdf_path).expanduser().resolve()),
            _package_dirs_arg(package_dir),
        )
        self.data = self.model.createData()
        self.ee_frame = ee_frame
        self.ee_frame_id = frame_id(self.model, ee_frame)
        self.active_joint_names = tuple(active_joint_names)
        self.active_mask = active_velocity_mask(self.model, self.active_joint_names)
        if q0 is not None:
            self.q = clip_to_limits(self.model, q0)
        elif initial_joint_values is not None:
            self.q = joint_vector_from_names(self.model, self.active_joint_names, initial_joint_values)
        else:
            self.q = home_configuration(self.model)

    def update_kinematics(self, q: np.ndarray | None = None) -> pin.SE3:
        if q is not None:
            self.q = clip_to_limits(self.model, q)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id]

    def forward_pose6(self, q: np.ndarray | None = None) -> np.ndarray:
        return se3_to_pose6(self.update_kinematics(q))

    def solve_transform(
        self,
        target: pin.SE3,
        max_iterations: int = 250,
        tolerance: float = 1e-4,
        damping: float = 1e-4,
        dt: float = 0.45,
        max_velocity: float = 0.35,
        position_weight: float = 1.0,
        rotation_weight: float = 0.55,
    ) -> IKResult:
        weights = np.array(
            [position_weight, position_weight, position_weight, rotation_weight, rotation_weight, rotation_weight],
            dtype=np.float64,
        )
        identity6 = np.eye(6)
        last_iterations = 0
        for iteration in range(1, max_iterations + 1):
            current = self.update_kinematics()
            error = pin.log6(current.actInv(target)).vector
            weighted_error = weights * error
            weighted_norm = float(np.linalg.norm(weighted_error))
            result = IKResult(
                self.q.copy(),
                bool(weighted_norm < tolerance),
                iteration - 1,
                float(np.linalg.norm(error[:3])),
                float(np.linalg.norm(error[3:])),
                weighted_norm,
            )
            if result.converged:
                return result
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
            last_iterations = iteration

        current = self.update_kinematics()
        error = pin.log6(current.actInv(target)).vector
        weighted_error = weights * error
        weighted_norm = float(np.linalg.norm(weighted_error))
        return IKResult(
            self.q.copy(),
            bool(weighted_norm < tolerance),
            last_iterations,
            float(np.linalg.norm(error[:3])),
            float(np.linalg.norm(error[3:])),
            weighted_norm,
        )

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
        return self.solve_transform(
            pose6_to_se3(target_pose),
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            dt=dt,
            max_velocity=max_velocity,
            position_weight=position_weight,
            rotation_weight=rotation_weight,
        )

    def trace_points(self, q: np.ndarray, link_names: tuple[str, ...]) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        points = []
        for name in link_names:
            result = self.model.getFrameId(name)
            if result < len(self.model.frames) and self.model.frames[result].name == name:
                points.append(self.data.oMf[result].translation.copy())
        return np.asarray(points, dtype=np.float64)


class PinkReplayIK:
    def __init__(
        self,
        fk: PinocchioIK,
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 0.5,
        posture_cost: float = 1e-3,
        dt: float = 1.0,
        solver: str = "quadprog",
        max_iters: int = 40,
        pos_tol: float = 5e-4,
        rot_tol: float = 5e-3,
        sub_step_pos: float = 0.01,
        sub_step_rot: float = np.deg2rad(6.0),
        divergence_pos: float = 0.05,
        divergence_rot: float = np.deg2rad(30.0),
    ) -> None:
        if pink is None or FrameTask is None or PostureTask is None:
            raise ImportError("未安装 pink 依赖，请先执行 `pip install -r requirements-ik.txt`。")
        self.fk = fk
        self.configuration = pink.Configuration(
            fk.model,
            fk.model.createData(),
            fk.q.copy(),
        )
        self.frame_task = FrameTask(
            fk.ee_frame,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1e-6,
        )
        self.posture_task = PostureTask(cost=posture_cost)
        self.posture_task.set_target(fk.q.copy())
        self.tasks = [self.frame_task, self.posture_task]
        self.dt = dt
        self.solver = solver
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.sub_step_pos = sub_step_pos
        self.sub_step_rot = sub_step_rot
        self.divergence_pos = divergence_pos
        self.divergence_rot = divergence_rot

    def _current_transform(self) -> pin.SE3:
        return self.configuration.get_transform_frame_to_world(self.fk.ee_frame)

    def _error_to_pose(self, target_pose: np.ndarray) -> np.ndarray:
        target = pose6_to_se3(target_pose)
        return pin.log6(self._current_transform().actInv(target)).vector

    def _subtarget_poses(self, target_pose: np.ndarray) -> list[np.ndarray]:
        current_pose = se3_to_pose6(self._current_transform())
        error = self._error_to_pose(target_pose)
        steps = max(
            1,
            int(
                np.ceil(
                    max(
                        np.linalg.norm(error[:3]) / max(self.sub_step_pos, 1e-9),
                        np.linalg.norm(error[3:]) / max(self.sub_step_rot, 1e-9),
                    )
                )
            ),
        )
        return [interpolate_pose6(current_pose, target_pose, step / steps) for step in range(1, steps + 1)]

    def solve(self, target_pose: np.ndarray) -> IKResult:
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(6)
        q_before = self.configuration.q.copy()
        total_iterations = 0
        for subtarget_pose in self._subtarget_poses(target_pose):
            subtarget = pose6_to_se3(subtarget_pose)
            self.frame_task.set_target(subtarget)
            for _iteration in range(1, self.max_iters + 1):
                total_iterations += 1
                velocity = pink.solve_ik(
                    self.configuration,
                    self.tasks,
                    self.dt,
                    solver=self.solver,
                )
                velocity[~self.fk.active_mask] = 0.0
                self.configuration.integrate_inplace(velocity, self.dt)
                self.configuration.update(self.configuration.q)
                error = pin.log6(self._current_transform().actInv(subtarget)).vector
                if np.linalg.norm(error[:3]) < self.pos_tol and np.linalg.norm(error[3:]) < self.rot_tol:
                    break

        error = self._error_to_pose(target_pose)
        position_error = float(np.linalg.norm(error[:3]))
        rotation_error = float(np.linalg.norm(error[3:]))
        converged = position_error < self.pos_tol and rotation_error < self.rot_tol
        if position_error > self.divergence_pos or rotation_error > self.divergence_rot:
            self.configuration.update(q_before)
            self.fk.q = q_before.copy()
            return IKResult(q_before.copy(), False, total_iterations, position_error, rotation_error)
        q_solution = clip_to_limits(self.fk.model, self.configuration.q.copy())
        if not np.allclose(q_solution, self.configuration.q):
            self.configuration.update(q_solution)
        self.fk.q = q_solution.copy()
        return IKResult(q_solution.copy(), converged, total_iterations, position_error, rotation_error)


def solve_ik_sequence(
    ik: PinocchioIK,
    poses: np.ndarray,
    *,
    ik_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, list[IKResult]]:
    solver = PinkReplayIK(ik, **(ik_options or {}))
    qs = []
    results = []
    for pose in poses:
        result = solver.solve(pose)
        results.append(result)
        qs.append(solver.configuration.q.copy())
        solver.posture_task.set_target(solver.configuration.q.copy())
        ik.q = solver.configuration.q.copy()
    return np.asarray(qs, dtype=np.float64), results
