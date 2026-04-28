#!/usr/bin/env python3
"""Piper 手动 IK 控制网页与 MeshCat 可视化入口。"""

from __future__ import annotations

import argparse
import atexit
import html
import json
import math
import socket
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import meshcat
    import meshcat.geometry as g
    import numpy as np
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
    from piper_ik_utils import PinocchioIK, clip_to_limits, joint_id
except ModuleNotFoundError as exc:
    print(f"Missing Python dependency: {exc.name}")
    print("Install dependencies with: python3 -m pip install -r requirements-ik.txt")
    raise SystemExit(1) from exc


DEFAULT_ACTIVE_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    corrected = u @ vt
    if np.linalg.det(corrected) < 0.0:
        u[:, -1] *= -1.0
        corrected = u @ vt
    return corrected


def yaw_pitch_roll(rotation: np.ndarray) -> tuple[float, float, float]:
    # ZYX decomposition: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    if abs(rotation[2, 0]) < 1.0 - 1e-9:
        pitch = math.asin(-rotation[2, 0])
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.pi / 2.0 if rotation[2, 0] <= -1.0 else -math.pi / 2.0
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return yaw, pitch, roll


def make_axes(length: float = 0.09) -> g.LineSegments:
    points = np.array(
        (
            (0.0, 0.0, 0.0),
            (length, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, length, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, length),
        )
    ).T
    colors = np.array(
        (
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.7, 0.2),
            (0.0, 0.7, 0.2),
            (0.0, 0.25, 1.0),
            (0.0, 0.25, 1.0),
        )
    ).T
    return g.LineSegments(
        g.PointsGeometry(points, color=colors),
        g.LineBasicMaterial(vertexColors=True, linewidth=4),
    )


def pose_to_dict(transform: pin.SE3) -> dict[str, Any]:
    yaw, pitch, roll = yaw_pitch_roll(transform.rotation)
    return {
        "x": float(transform.translation[0]),
        "y": float(transform.translation[1]),
        "z": float(transform.translation[2]),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
    }


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    raw = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def render_html_template(template_name: str, **values: str) -> str:
    body = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    for key, value in values.items():
        body = body.replace(f"{{{{ {key} }}}}", html.escape(str(value), quote=True))
    return body


@dataclass
class SolverResult:
    converged: bool = False
    iterations: int = 0
    weighted_error: float = math.inf
    position_error: float = math.inf
    rotation_error: float = math.inf


class PiperIKApp:
    def __init__(
        self,
        urdf_path: Path,
        package_dirs: list[Path],
        ee_frame: str,
        active_joints: tuple[str, ...],
        tolerance: float,
        max_iterations: int,
        dt: float,
        damping: float,
        max_velocity: float,
        position_weight: float,
        rotation_weight: float,
    ) -> None:
        self.urdf_path = urdf_path
        self.package_dirs = package_dirs
        self.ee_frame_name = ee_frame
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.dt = dt
        self.damping = damping
        self.max_velocity = max_velocity
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.lock = threading.RLock()
        self.ik = PinocchioIK(
            urdf_path=urdf_path,
            package_dir=package_dirs,
            ee_frame=ee_frame,
            active_joint_names=active_joints,
        )
        self.model = self.ik.model
        self.data = self.ik.data
        self.ee_frame_id = self.ik.ee_frame_id
        self.q_home = self.ik.q.copy()
        self.last_result = SolverResult()
        self.viz: MeshcatVisualizer | None = None
        self.meshcat_server: subprocess.Popen | None = None
        self.meshcat_url = ""
        self.markers_enabled = False
        self._update_kinematics()
        current = self.data.oMf[self.ee_frame_id]
        self.target = pin.SE3(current.rotation.copy(), current.translation.copy())

    @property
    def q(self) -> np.ndarray:
        return self.ik.q

    @q.setter
    def q(self, value: np.ndarray) -> None:
        self.ik.q = clip_to_limits(self.model, value)

    def _joint_id(self, name: str) -> int:
        return joint_id(self.model, name)

    def _update_kinematics(self) -> None:
        self.ik.update_kinematics()

    def init_viewer(self, root_node: str, open_meshcat: bool, meshcat_port: int | None) -> None:
        with self.lock:
            self.viz = MeshcatVisualizer(self.model, self.ik.collision_model, self.ik.visual_model)
            viewer = None
            if meshcat_port is not None:
                viewer, self.meshcat_server = make_meshcat_viewer(meshcat_port)
            self.viz.initViewer(viewer=viewer, open=open_meshcat)
            self.viz.loadViewerModel(rootNodeName=root_node)
            url_fn = getattr(self.viz.viewer, "url", None)
            self.meshcat_url = url_fn() if callable(url_fn) else ""
            try:
                self.viz.viewer["piper_ik/target_axes"].set_object(make_axes(0.105))
                self.viz.viewer["piper_ik/current_axes"].set_object(make_axes(0.075))
                self.viz.viewer["piper_ik/target_tip"].set_object(
                    g.Sphere(0.014),
                    g.MeshLambertMaterial(color=0x12a150, transparent=True, opacity=0.75),
                )
                self.markers_enabled = True
            except Exception as exc:  # MeshCat marker support should not block IK use.
                print(f"Warning: failed to create MeshCat target markers: {exc}")
                self.markers_enabled = False
            self.display()

    def display(self) -> None:
        if self.viz is None:
            return
        self._update_kinematics()
        self.viz.display(self.q)
        if not self.markers_enabled:
            return
        current = self.data.oMf[self.ee_frame_id]
        self.viz.viewer["piper_ik/target_axes"].set_transform(self.target.homogeneous)
        self.viz.viewer["piper_ik/target_tip"].set_transform(self.target.homogeneous)
        self.viz.viewer["piper_ik/current_axes"].set_transform(current.homogeneous)

    def step_translation(self, axis: str, amount: float) -> SolverResult:
        with self.lock:
            index = {"x": 0, "y": 1, "z": 2}[axis]
            translation = self.target.translation.copy()
            translation[index] += amount
            self.target = pin.SE3(self.target.rotation.copy(), translation)
            return self.solve_and_display()

    def step_rotation(self, axis: str, amount: float) -> SolverResult:
        with self.lock:
            if axis == "yaw":
                delta = rot_z(amount)
            elif axis == "pitch":
                delta = rot_y(amount)
            elif axis == "roll":
                delta = rot_x(amount)
            else:
                raise ValueError(f"Unsupported rotation axis: {axis}")
            rotation = orthonormalize(delta @ self.target.rotation)
            self.target = pin.SE3(rotation, self.target.translation.copy())
            return self.solve_and_display()

    def home(self) -> SolverResult:
        with self.lock:
            self.q = self.q_home.copy()
            self._update_kinematics()
            current = self.data.oMf[self.ee_frame_id]
            self.target = pin.SE3(current.rotation.copy(), current.translation.copy())
            self.last_result = SolverResult(True, 0, 0.0, 0.0, 0.0)
            self.display()
            return self.last_result

    def reset_target_to_current(self) -> SolverResult:
        with self.lock:
            self._update_kinematics()
            current = self.data.oMf[self.ee_frame_id]
            self.target = pin.SE3(current.rotation.copy(), current.translation.copy())
            self.last_result = SolverResult(True, 0, 0.0, 0.0, 0.0)
            self.display()
            return self.last_result

    def solve_and_display(self) -> SolverResult:
        self.last_result = self.solve()
        self.display()
        return self.last_result

    def solve(self) -> SolverResult:
        result = self.ik.solve_transform(
            self.target,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            damping=self.damping,
            dt=self.dt,
            max_velocity=self.max_velocity,
            position_weight=self.position_weight,
            rotation_weight=self.rotation_weight,
        )
        self._update_kinematics()
        return SolverResult(
            converged=result.converged,
            iterations=result.iterations,
            weighted_error=result.weighted_error,
            position_error=result.position_error,
            rotation_error=result.rotation_error,
        )

    def state(self) -> dict[str, Any]:
        with self.lock:
            self._update_kinematics()
            current = self.data.oMf[self.ee_frame_id]
            joint_values: dict[str, float] = {}
            for joint_name in DEFAULT_ACTIVE_JOINTS:
                joint_id = self._joint_id(joint_name)
                joint = self.model.joints[joint_id]
                joint_values[joint_name] = float(self.q[joint.idx_q])
            return {
                "ee_frame": self.ee_frame_name,
                "meshcat_url": self.meshcat_url,
                "target": pose_to_dict(self.target),
                "current": pose_to_dict(current),
                "result": self.last_result.__dict__,
                "joints": joint_values,
            }


class ControlHandler(BaseHTTPRequestHandler):
    app: PiperIKApp

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
            if path == "/api/step":
                kind = payload["kind"]
                axis = payload["axis"]
                amount = float(payload["amount"])
                if kind == "translate":
                    result = self.app.step_translation(axis, amount)
                elif kind == "rotate":
                    result = self.app.step_rotation(axis, amount)
                else:
                    raise ValueError(f"Unsupported step kind: {kind}")
                json_response(self, {"ok": True, "result": result.__dict__, "state": self.app.state()})
            elif path == "/api/home":
                result = self.app.home()
                json_response(self, {"ok": True, "result": result.__dict__, "state": self.app.state()})
            elif path == "/api/reset-target":
                result = self.app.reset_target_to_current()
                json_response(self, {"ok": True, "result": result.__dict__, "state": self.app.state()})
            else:
                json_response(self, {"ok": False, "error": "Not found"}, 404)
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc)}, 400)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def index_html(meshcat_url: str) -> str:
    return render_html_template("piper_ik_visualizer.html", meshcat_url=meshcat_url)


def make_server(host: str, port: int, handler: type[ControlHandler]) -> ThreadingHTTPServer:
    try:
        return ThreadingHTTPServer((host, port), handler)
    except OSError:
        if port == 0:
            raise
        print(f"Control port {port} is busy; falling back to an available port.")
        return ThreadingHTTPServer((host, 0), handler)


def port_available(host: str, port: int) -> bool:
    probes = {host, "127.0.0.1", "0.0.0.0"}
    for probe_host in probes:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((probe_host, port))
            except OSError:
                return False
    return True


def choose_port(host: str, preferred_port: int, attempts: int = 1000) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        if port_available(host, port):
            return port
    raise OSError(f"No available TCP port found from {preferred_port} to {preferred_port + attempts - 1}.")


def make_meshcat_viewer(preferred_port: int) -> tuple[meshcat.Visualizer, subprocess.Popen]:
    port = choose_port("127.0.0.1", preferred_port)
    if port != preferred_port:
        print(f"MeshCat port {preferred_port} is busy; using {port}.")
    code = (
        "from meshcat.servers.zmqserver import ZMQWebSocketBridge\n"
        f"bridge = ZMQWebSocketBridge(port={port})\n"
        "print('zmq_url={:s}'.format(bridge.zmq_url), flush=True)\n"
        "print('web_url={:s}'.format(bridge.web_url), flush=True)\n"
        "bridge.run()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    zmq_url = ""
    web_url = ""
    while not zmq_url or not web_url:
        line = process.stdout.readline() if process.stdout else ""
        if not line:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    "MeshCat server exited before startup completed.\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            continue
        line = line.strip()
        if line.startswith("zmq_url="):
            zmq_url = line.split("=", 1)[1]
        elif line.startswith("web_url="):
            web_url = line.split("=", 1)[1]
    viewer = meshcat.Visualizer(zmq_url=zmq_url)
    viewer.window.web_url = web_url

    def cleanup() -> None:
        if process.poll() is None:
            process.terminate()

    atexit.register(cleanup)
    return viewer, process


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    default_urdf = root / "piper_description" / "urdf" / "piper_description.urdf"
    parser = argparse.ArgumentParser(description="Piper Pinocchio IK visualizer with MeshCat controls.")
    parser.add_argument("--urdf", type=Path, default=default_urdf, help="URDF path.")
    parser.add_argument(
        "--package-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing the piper_description package. Defaults to this repo root.",
    )
    parser.add_argument("--ee-frame", default="gripper_base", help="End-effector frame/link name.")
    parser.add_argument(
        "--active-joints",
        default=",".join(DEFAULT_ACTIVE_JOINTS),
        help="Comma-separated joints used by IK. Empty means all velocity dimensions.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Control panel host.")
    parser.add_argument("--control-port", type=int, default=8010, help="Control panel port.")
    parser.add_argument("--meshcat-port", type=int, default=7050, help="Preferred MeshCat web port.")
    parser.add_argument("--root-node", default="piper", help="MeshCat robot root node.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Weighted IK convergence tolerance.")
    parser.add_argument("--max-iters", type=int, default=80, help="Max IK iterations per UI step.")
    parser.add_argument("--dt", type=float, default=0.45, help="Integration step scale.")
    parser.add_argument("--damping", type=float, default=1e-4, help="Damped least-squares coefficient.")
    parser.add_argument("--max-velocity", type=float, default=0.35, help="Max velocity norm per iteration.")
    parser.add_argument("--position-weight", type=float, default=1.0, help="Translation error weight.")
    parser.add_argument("--rotation-weight", type=float, default=0.55, help="Rotation error weight.")
    parser.add_argument("--open-meshcat", action="store_true", help="Also open the raw MeshCat viewer.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the control panel in a browser.")
    return parser


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args()
    urdf_path = args.urdf.expanduser().resolve()
    package_dirs = [path.expanduser().resolve() for path in args.package_dir] if args.package_dir else [root]
    active_joints = tuple(name.strip() for name in args.active_joints.split(",") if name.strip())

    app = PiperIKApp(
        urdf_path=urdf_path,
        package_dirs=package_dirs,
        ee_frame=args.ee_frame,
        active_joints=active_joints,
        tolerance=args.tol,
        max_iterations=args.max_iters,
        dt=args.dt,
        damping=args.damping,
        max_velocity=args.max_velocity,
        position_weight=args.position_weight,
        rotation_weight=args.rotation_weight,
    )
    app.init_viewer(root_node=args.root_node, open_meshcat=args.open_meshcat, meshcat_port=args.meshcat_port)

    ControlHandler.app = app
    server = make_server(args.host, args.control_port, ControlHandler)
    host, port = server.server_address[:2]
    control_url = f"http://{host}:{port}/"
    print(f"URDF: {urdf_path}")
    print(f"Package dirs: {', '.join(str(path) for path in package_dirs)}")
    print(f"End-effector frame: {args.ee_frame}")
    print(f"MeshCat: {app.meshcat_url or '(MeshCat URL unavailable)'}")
    print(f"Control panel: {control_url}")
    if not args.no_open:
        webbrowser.open(control_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
