#!/usr/bin/env python3
"""订阅 LCP WebSocket ``state``，将左右臂位姿写入日志。

``--mode mapped``（默认）按 ``livumi_service.umi_teleop_controller`` 的当前逻辑输出
机器人目标位姿：

- 建立 UMI 基准帧：每只手首次收到有效 ``state`` 位姿时锁定；
- 建立机器人基准帧：首次可获取到 B 端 ``arm_ee_poses`` 时锁定；
- 后续每一帧计算 ``delta = current_umi - base_umi``；
- 最终输出 ``_apply_delta_to_current_pose(base_arm, delta)``，即与 ``move_arms`` 同语义的目标值。

在仓库根目录执行::

    PYTHONPATH=livumi_service python scripts/log_umi_arms_state.py --log-file /tmp/umi_arms.log

可选 ``--interval`` 节流（秒）。``--mode umi`` 仅打原始 A 端 UMI；``--mode both`` 两者都打。
``--poll-robot-b`` 会额外打印当前 B 端 ``arm_ee_poses`` 反馈；而 ``mapped`` 模式本身就需要访问 B 端来锁定机器人基准帧。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import numpy as np
from scipy.spatial.transform import Rotation as R

# 仓库根 …/lcp_1：便于 ``import lcp_ws_client`` 与 ``import livumi_service``
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "livumi_service"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from lcp_ws_client.client import LcpStateWebSocketClient
from lcp_ws_client.config_environ import default_host_port_from_environ
from livumi_service.delta_pose import calculate_delta_pose
from livumi_service.state_parse import parse_arm_entry, pose6_livumi_to_pose7
from livumi_service.teleop_target_client import TeleopTargetClient
from livumi_service.umi_teleop_controller import (
    LEFT_TRANS_MATRIX,
    RIGHT_TRANS_MATRIX,
    UmiTeleopController,
    _apply_delta_deadband,
    _normalize_arm_pose_vec,
)

_TRANS_BY_HAND: dict[str, np.ndarray] = {"left": LEFT_TRANS_MATRIX, "right": RIGHT_TRANS_MATRIX}


def _cmd_pose6_to_pose7(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    rot = R.from_euler("ZYX", [pose6[5], pose6[4], pose6[3]], degrees=False)
    return np.concatenate([pose6[:3], rot.as_quat()])


class ControllerStylePoseMapper:
    def __init__(self, client: TeleopTargetClient) -> None:
        self._client = client
        self._controller = UmiTeleopController(client)

    def _maybe_lock_arm_bases(self) -> None:
        lp, rp = self._client.get_arm_ee_poses()
        poses = {
            "left": _normalize_arm_pose_vec(lp),
            "right": _normalize_arm_pose_vec(rp),
        }
        for hand, pose in poses.items():
            if self._controller.last_arm_poses[hand] is None and pose is not None:
                self._controller.last_arm_poses[hand] = pose

    def format_robot_target_line(self, arms: dict[str, Any]) -> str:
        try:
            self._maybe_lock_arm_bases()
        except Exception as e:
            return f"[livumi] robot_target(同 umi_teleop_controller) <机器人基准帧拉取失败: {e}>"

        parts: list[str] = []
        for hand in ("left", "right"):
            arm = arms.get(hand)
            pose6, grip, ts = parse_arm_entry(arm)
            if pose6 is None:
                parts.append(f"{hand}=<无位姿>")
                continue

            try:
                cur7 = pose6_livumi_to_pose7(pose6)
            except Exception as e:
                parts.append(f"{hand}=<pose6->pose7失败 {e}>")
                continue
            self._controller.current_umi_pose_7d[hand] = cur7.copy()

            base_arm = self._controller.last_arm_poses[hand]
            if base_arm is None:
                parts.append(f"{hand}=<无机器人基准帧> g={grip:.3f} ts={ts:.3f}")
                continue

            base_umi = self._controller.last_umi_pose_7d[hand]
            if base_umi is None:
                self._controller.last_umi_pose_7d[hand] = cur7.copy()
                delta = np.zeros(7, dtype=np.float64)
            else:
                delta = _apply_delta_deadband(calculate_delta_pose(base_umi, cur7))

            pose_out = self._controller._apply_delta_to_current_pose(
                _TRANS_BY_HAND[hand], base_arm, delta, grip
            )
            p7_out = _cmd_pose6_to_pose7(pose_out[:6])
            parts.append(
                f"{hand}=[{_fmt_pose6(pose_out[:6])}] g={grip:.3f} ts={ts:.3f}"
                f" pos+quat(7)=[{_fmt_p7(p7_out)}]"
            )

        return "[livumi] robot_target(同 umi_teleop_controller) " + " | ".join(parts)


def _fmt_pose6(p6: np.ndarray) -> str:
    return ",".join(f"{float(x):.4f}" for x in p6)


def _fmt_p7(p7: np.ndarray) -> str:
    return ",".join(f"{float(x):.4f}" for x in p7)


def _format_umi_raw_line(arms: dict[str, Any]) -> str:
    parts: list[str] = []
    for hand in ("left", "right"):
        arm = arms.get(hand)
        pose6, grip, ts = parse_arm_entry(arm)
        if pose6 is None:
            parts.append(f"{hand}=<无位姿>")
            continue
        p6s = _fmt_pose6(pose6)
        try:
            p7 = pose6_livumi_to_pose7(pose6)
            extra = f" pos+quat(7)=[{_fmt_p7(p7)}]"
        except Exception:
            extra = ""
        parts.append(f"{hand}=[{p6s}] g={grip:.3f} ts={ts:.3f}{extra}")
    return "[livumi] A/state UMI(raw) " + " | ".join(parts)


def _format_b_poses_line(fetch: Callable[[], tuple[Optional[np.ndarray], Optional[np.ndarray]]]) -> str:
    try:
        lp, rp = fetch()
    except Exception as e:
        return f"[livumi] B/arm_ee_poses <拉取失败: {e}>"

    def one(name: str, p: Optional[np.ndarray]) -> str:
        if p is None or p.size < 6:
            return f"{name}=<无>"
        p6 = np.asarray(p, dtype=np.float64).reshape(-1)[:6]
        return f"{name}=[{_fmt_pose6(p6)}]"

    return "[livumi] B/arm_ee_poses(机器人反馈) " + " | ".join([one("left", lp), one("right", rp)])


def main() -> int:
    env_host, env_port = default_host_port_from_environ()
    ap = argparse.ArgumentParser(description="UMI state 日志：原始 A 端或按 controller 计算的机器人目标位姿，可选 B 端末端")
    ap.add_argument(
        "--log-file",
        default="umi_arms_state.log",
        help="输出日志路径（默认当前目录 umi_arms_state.log）",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="节流间隔（秒），<=0 表示每帧 state 都写",
    )
    ap.add_argument(
        "--mode",
        choices=("mapped", "umi", "both"),
        default="mapped",
        help="mapped=按 umi_teleop_controller 输出机器人目标位姿(默认); umi=仅 A 端原始; both=两行各打一次",
    )
    ap.add_argument(
        "--poll-robot-b",
        action="store_true",
        help="同节拍额外请求 B 端 internal/teleop arm_ee_poses（mapped 本身也会取一次基准帧）",
    )
    ap.add_argument("--host", default=env_host, help="LCP 后端地址")
    ap.add_argument("--port", type=int, default=env_port, help="LCP API 端口")
    ap.add_argument("--tls", action="store_true", help="使用 wss://")
    args = ap.parse_args()

    log = logging.getLogger("umi_arms_state_file")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(args.log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    log.propagate = False

    http_cm: Optional[httpx.Client] = None
    teleop_client: Optional[TeleopTargetClient] = None
    mapper: Optional[ControllerStylePoseMapper] = None
    if args.mode in ("mapped", "both") or args.poll_robot_b:
        http_cm = httpx.Client()
        teleop_client = TeleopTargetClient(http_cm)
    if teleop_client is not None:
        mapper = ControllerStylePoseMapper(teleop_client)

    last_mono = 0.0
    interval = float(args.interval)

    def on_state(data: dict[str, Any]) -> None:
        nonlocal last_mono
        arms = data.get("arms") or {}
        if not isinstance(arms, dict):
            return
        now = time.monotonic()
        if interval > 0 and now - last_mono < interval:
            return
        last_mono = now
        if args.mode in ("umi", "both"):
            log.info("%s", _format_umi_raw_line(arms))
        if args.mode in ("mapped", "both"):
            if mapper is None:
                log.info("[livumi] robot_target(同 umi_teleop_controller) <未初始化 TeleopTargetClient>")
            else:
                log.info("%s", mapper.format_robot_target_line(arms))
        if args.poll_robot_b and teleop_client is not None:
            log.info("%s", _format_b_poses_line(teleop_client.get_arm_ee_poses))

    client = LcpStateWebSocketClient(
        args.host,
        args.port,
        topics=["state"],
        use_tls=args.tls,
        on_state=on_state,
    )
    client.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        if http_cm is not None:
            try:
                http_cm.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
