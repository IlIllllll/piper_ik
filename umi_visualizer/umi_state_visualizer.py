#!/usr/bin/env python3
"""UMI 左右臂日志可视化页面服务。"""

from __future__ import annotations

import argparse
import base64
from collections import deque
import hashlib
import html
import json
import re
import socket
import struct
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "templates"
DEFAULT_SAMPLE_LOG = ROOT / "sample_umi_arms_state.log"
DEFAULT_WS_URL = "ws://192.168.150.27:8100/ws"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

LINE_RE = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+"
    r"(?P<level>[A-Z]+)\s+(?P<body>.*)$"
)
GRIP_RE = re.compile(r"\bg=(?P<value>[-+]?\d+(?:\.\d+)?)")
SOURCE_TS_RE = re.compile(r"\bts=(?P<value>[-+]?\d+(?:\.\d+)?)")
POSE7_RE = re.compile(r"pos\+quat\(7\)=\[(?P<value>[^\]]+)\]")

KIND_TITLES = {
    "robot_target": "robot_target",
    "umi_raw": "UMI raw",
    "robot_feedback": "B feedback",
}


def _float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _parse_time(stamp: str | None) -> tuple[str | None, float | None]:
    if not stamp:
        return None, None
    parsed = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f")
    return parsed.isoformat(timespec="milliseconds"), parsed.timestamp()


def _iso_now() -> tuple[str, float]:
    now = time.time()
    return datetime.fromtimestamp(now).isoformat(timespec="milliseconds"), now


def _classify_body(body: str) -> tuple[str | None, str]:
    if "[livumi] A/state UMI(raw)" in body:
        return "umi_raw", body.split("[livumi] A/state UMI(raw)", 1)[1].strip()
    if "[livumi] robot_target" in body:
        marker_end = body.find(")")
        return "robot_target", body[marker_end + 1 :].strip() if marker_end >= 0 else body
    if "[livumi] B/arm_ee_poses" in body:
        marker_end = body.find(")")
        return "robot_feedback", body[marker_end + 1 :].strip() if marker_end >= 0 else body
    return None, body


def _hand_parts(text: str) -> list[str]:
    starts = [index for index in (text.find("left="), text.find("right=")) if index >= 0]
    if not starts:
        return []
    return [part.strip() for part in re.split(r"\s+\|\s+", text[min(starts) :]) if part.strip()]


def _parse_hand_part(part: str) -> tuple[str | None, dict[str, Any]]:
    if "=" not in part:
        return None, {"valid": False, "message": part}
    hand, rest = part.split("=", 1)
    hand = hand.strip()
    rest = rest.strip()
    if hand not in ("left", "right"):
        return None, {"valid": False, "message": part}
    if rest.startswith("<"):
        return hand, {"valid": False, "message": rest.strip("<>")}

    pose_match = re.match(r"\[(?P<pose>[^\]]+)\](?P<tail>.*)$", rest)
    if not pose_match:
        return hand, {"valid": False, "message": rest}

    tail = pose_match.group("tail")
    pose6 = _float_list(pose_match.group("pose"))
    grip_match = GRIP_RE.search(tail)
    ts_match = SOURCE_TS_RE.search(tail)
    pose7_match = POSE7_RE.search(tail)
    return hand, {
        "valid": len(pose6) >= 6,
        "pose6": pose6[:6],
        "grip": None if grip_match is None else float(grip_match.group("value")),
        "source_ts": None if ts_match is None else float(ts_match.group("value")),
        "pose7": None if pose7_match is None else _float_list(pose7_match.group("value"))[:7],
    }


def _sample_from_log_line(raw: str, index: int = 0) -> dict[str, Any] | None:
    match = LINE_RE.match(raw.strip())
    stamp = match.group("stamp") if match else None
    level = match.group("level") if match else "INFO"
    body = match.group("body") if match else raw.strip()
    kind, payload = _classify_body(body)
    if kind is None and ("left=" in payload or "right=" in payload):
        kind = "umi_raw"
    if kind is None:
        return None

    iso_time, epoch = _parse_time(stamp)
    if iso_time is None or epoch is None:
        iso_time, epoch = _iso_now()
    hands: dict[str, dict[str, Any]] = {}
    for part in _hand_parts(payload):
        hand, parsed = _parse_hand_part(part)
        if hand:
            hands[hand] = parsed
    return {
        "index": index,
        "line": index + 1,
        "time": iso_time,
        "epoch": epoch,
        "level": level,
        "kind": kind,
        "kind_title": KIND_TITLES[kind],
        "hands": hands,
        "transport": "live",
    }


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _pose_from_mapping(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)):
        if len(value) >= 6:
            return [float(item) for item in value[:6]]
        return None
    if not isinstance(value, dict):
        return None

    direct = _first_present(
        value,
        (
            "pose6",
            "pose",
            "ee_pose",
            "end_effector_pose",
            "target_pose",
            "arm_pose",
            "data",
        ),
    )
    if isinstance(direct, (list, tuple)) and len(direct) >= 6:
        return [float(item) for item in direct[:6]]

    if all(key in value for key in ("x", "y", "z")):
        return [
            float(value["x"]),
            float(value["y"]),
            float(value["z"]),
            float(_first_present(value, ("roll", "rx", "r")) or 0.0),
            float(_first_present(value, ("pitch", "ry", "p")) or 0.0),
            float(_first_present(value, ("yaw", "rz", "y")) or 0.0),
        ]
    return None


def _hand_from_json(value: Any) -> dict[str, Any]:
    pose6 = _pose_from_mapping(value)
    grip = None
    source_ts = None
    pose7 = None
    if isinstance(value, dict):
        grip = _first_present(value, ("g", "grip", "gripper", "end_effector_value"))
        if isinstance(grip, (list, tuple)):
            grip = grip[0] if grip else None
        source_ts = _first_present(value, ("ts", "timestamp", "time"))
        p7 = _first_present(value, ("pose7", "pos_quat", "position_quaternion"))
        if isinstance(p7, (list, tuple)) and len(p7) >= 7:
            pose7 = [float(item) for item in p7[:7]]
    elif isinstance(value, (list, tuple)) and len(value) >= 7:
        grip = value[6]
        if len(value) >= 8:
            source_ts = value[7]
    return {
        "valid": pose6 is not None,
        "pose6": pose6,
        "grip": None if grip is None else float(grip),
        "source_ts": None if source_ts is None else float(source_ts),
        "pose7": pose7,
    }


def _sample_from_json(obj: Any, index: int = 0) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    data = obj.get("data") if isinstance(obj.get("data"), dict) else obj
    root = data.get("arms") if isinstance(data.get("arms"), dict) else data
    hands: dict[str, dict[str, Any]] = {}
    for hand in ("left", "right"):
        if hand in root:
            hands[hand] = _hand_from_json(root[hand])
    if not hands:
        return None

    kind = str(obj.get("kind") or data.get("kind") or obj.get("mode") or data.get("mode") or obj.get("type") or "umi_raw")
    if kind in ("raw", "umi", "state"):
        kind = "umi_raw"
    elif kind in ("mapped", "target", "robot_target"):
        kind = "robot_target"
    elif kind in ("feedback", "robot_feedback", "b"):
        kind = "robot_feedback"
    else:
        kind = "umi_raw"

    epoch_value = obj.get("timestamp") or data.get("timestamp") or obj.get("time") or data.get("time") or obj.get("ts") or data.get("ts")
    if epoch_value is None:
        epoch_value = next(
            (
                hand_data.get("source_ts")
                for hand_data in hands.values()
                if hand_data.get("source_ts") is not None
            ),
            None,
        )
    if isinstance(epoch_value, (int, float)):
        epoch = float(epoch_value)
        if epoch > 1e12:
            epoch /= 1000.0
        iso_time = datetime.fromtimestamp(epoch).isoformat(timespec="milliseconds")
    else:
        iso_time, epoch = _iso_now()

    return {
        "index": index,
        "line": index + 1,
        "time": iso_time,
        "epoch": epoch,
        "level": "INFO",
        "kind": kind,
        "kind_title": KIND_TITLES[kind],
        "hands": hands,
        "transport": "live",
    }


def parse_live_packet(payload: bytes, index: int = 0) -> dict[str, Any] | None:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError:
        parsed_json = None
    if parsed_json is not None:
        sample = _sample_from_json(parsed_json, index)
        if sample is not None:
            sample["raw"] = text[:500]
            return sample
    sample = _sample_from_log_line(text, index)
    if sample is not None:
        sample["raw"] = text[:500]
    return sample


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("WebSocket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_ws_frame(sock: socket.socket, opcode: int, payload: bytes = b"", *, masked: bool) -> None:
    first = 0x80 | (opcode & 0x0F)
    length = len(payload)
    header = bytearray([first])
    mask_bit = 0x80 if masked else 0
    if length < 126:
        header.append(mask_bit | length)
    elif length < (1 << 16):
        header.extend((mask_bit | 126, (length >> 8) & 0xFF, length & 0xFF))
    else:
        header.append(mask_bit | 127)
        header.extend(length.to_bytes(8, "big"))
    if masked:
        mask = struct.pack("!I", int(time.time_ns() & 0xFFFFFFFF))
        header.extend(mask)
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(bytes(header) + payload)


def _recv_ws_frame(sock: socket.socket) -> tuple[int, bytes]:
    first, second = _recv_exact(sock, 2)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = int.from_bytes(_recv_exact(sock, 2), "big")
    elif length == 127:
        length = int.from_bytes(_recv_exact(sock, 8), "big")
    mask = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, length) if length else b""
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return opcode, payload


def _websocket_accept_key(key: str) -> str:
    digest = hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _connect_ws(url: str, timeout: float = 5.0) -> socket.socket:
    parsed = urlparse(url)
    if parsed.scheme != "ws":
        raise ValueError("当前内置客户端只支持 ws://，不支持 wss://")
    host = parsed.hostname
    if not host:
        raise ValueError(f"WebSocket URL 缺少 host: {url}")
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)
    key = base64.b64encode(struct.pack("!QQ", time.time_ns(), id(sock))).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("WebSocket handshake closed")
        response.extend(chunk)
        if len(response) > 65536:
            raise ConnectionError("WebSocket handshake response too large")
    header = response.decode("iso-8859-1", errors="replace")
    if " 101 " not in header.split("\r\n", 1)[0]:
        raise ConnectionError(f"WebSocket handshake failed: {header.splitlines()[0] if header else '<empty>'}")
    expected = _websocket_accept_key(key)
    if expected.lower() not in header.lower():
        raise ConnectionError("WebSocket handshake accept key mismatch")
    sock.settimeout(0.5)
    return sock


def parse_umi_log(log_file: Path, max_points: int | None = None) -> dict[str, Any]:
    log_file = log_file.expanduser().resolve()
    if not log_file.is_file():
        raise FileNotFoundError(f"UMI 日志不存在: {log_file}")

    samples: list[dict[str, Any]] = []
    with log_file.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            sample = _sample_from_log_line(raw, len(samples))
            if sample is None:
                continue
            sample["line"] = line_no
            sample.pop("transport", None)
            samples.append(sample)

    if max_points is not None and max_points > 0 and len(samples) > max_points:
        last = len(samples) - 1
        indices = sorted({round(index * last / (max_points - 1)) for index in range(max_points)})
        samples = [samples[index] for index in indices]

    first_epoch = next((sample["epoch"] for sample in samples if sample["epoch"] is not None), None)
    for fallback_t, sample in enumerate(samples):
        sample["index"] = fallback_t
        sample["t"] = fallback_t if first_epoch is None or sample["epoch"] is None else sample["epoch"] - first_epoch

    return {
        "log_file": str(log_file),
        "source": "log",
        "count": len(samples),
        "kinds": {
            kind: sum(1 for sample in samples if sample["kind"] == kind)
            for kind in ("robot_target", "umi_raw", "robot_feedback")
        },
        "samples": samples,
    }


class LiveUmiReceiver:
    def __init__(
        self,
        *,
        bind_host: str,
        port: int,
        source_host: str | None,
        max_samples: int = 12000,
    ) -> None:
        self.bind_host = bind_host
        self.port = port
        self.source_host = source_host or None
        self.max_samples = max_samples
        self.samples: deque[dict[str, Any]] = deque(maxlen=max_samples)
        self.lock = threading.RLock()
        self.started_at = time.time()
        self.packet_count = 0
        self.drop_count = 0
        self.last_error: str | None = None
        self.last_addr: str | None = None
        self._stop = threading.Event()
        self._receiving = threading.Event()
        self._receiving.set()
        self._thread = threading.Thread(target=self._run, name="umi-udp-receiver", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._receiving.set()

    def set_receiving(self, enabled: bool) -> None:
        if enabled:
            self._receiving.set()
            return
        self._receiving.clear()

    def is_receiving(self) -> bool:
        return self._receiving.is_set()

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.5)
        try:
            sock.bind((self.bind_host, self.port))
        except Exception as exc:
            self.last_error = f"UDP bind failed: {exc}"
            sock.close()
            return
        try:
            while not self._stop.is_set():
                if not self._receiving.is_set():
                    self._receiving.wait(0.2)
                    continue
                try:
                    payload, addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                host, port = addr[:2]
                if self.source_host and host != self.source_host:
                    continue
                with self.lock:
                    self.packet_count += 1
                    index = self.packet_count - 1
                sample = parse_live_packet(payload, index=index)
                with self.lock:
                    self.last_addr = f"{host}:{port}"
                    if sample is None:
                        self.drop_count += 1
                        self.last_error = f"无法解析 UDP 数据: {payload[:120]!r}"
                        continue
                    sample["source_addr"] = self.last_addr
                    self.samples.append(sample)
                    self.last_error = None
        finally:
            sock.close()

    def snapshot(self, max_points: int | None = None) -> dict[str, Any]:
        with self.lock:
            samples = list(self.samples)
            packet_count = self.packet_count
            drop_count = self.drop_count
            last_error = self.last_error
            last_addr = self.last_addr
            receiving = self.is_receiving()
        if max_points is not None and max_points > 0 and len(samples) > max_points:
            last = len(samples) - 1
            indices = sorted({round(index * last / (max_points - 1)) for index in range(max_points)})
            samples = [samples[index] for index in indices]
        first_epoch = next((sample["epoch"] for sample in samples if sample["epoch"] is not None), None)
        for fallback_t, sample in enumerate(samples):
            sample["index"] = fallback_t
            sample["t"] = fallback_t if first_epoch is None or sample["epoch"] is None else sample["epoch"] - first_epoch
        return {
            "log_file": f"udp://{self.bind_host}:{self.port}",
            "source": "live_udp",
            "source_filter": self.source_host,
            "receiving": receiving,
            "last_addr": last_addr,
            "count": len(samples),
            "packet_count": packet_count,
            "drop_count": drop_count,
            "last_error": last_error,
            "kinds": {
                kind: sum(1 for sample in samples if sample["kind"] == kind)
                for kind in ("robot_target", "umi_raw", "robot_feedback")
            },
            "samples": samples,
        }


class WebSocketUmiReceiver:
    def __init__(
        self,
        *,
        url: str,
        topics: Iterable[str] = ("state",),
        max_samples: int = 12000,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.url = url
        self.topics = list(topics)
        self.max_samples = max_samples
        self.reconnect_delay = reconnect_delay
        self.samples: deque[dict[str, Any]] = deque(maxlen=max_samples)
        self.lock = threading.RLock()
        self.started_at = time.time()
        self.packet_count = 0
        self.drop_count = 0
        self.reconnect_count = 0
        self.connected = False
        self.last_error: str | None = None
        self.last_addr: str | None = None
        self._stop = threading.Event()
        self._receiving = threading.Event()
        self._receiving.set()
        self._sock: socket.socket | None = None
        self._thread = threading.Thread(target=self._run, name="umi-websocket-receiver", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._receiving.set()
        self._close_active_socket()

    def set_receiving(self, enabled: bool) -> None:
        if enabled:
            self._receiving.set()
            return
        self._receiving.clear()
        self._close_active_socket()

    def is_receiving(self) -> bool:
        return self._receiving.is_set()

    def _close_active_socket(self) -> None:
        with self.lock:
            sock = self._sock
            self._sock = None
            self.connected = False
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._receiving.is_set():
                self._receiving.wait(0.2)
                continue
            sock: socket.socket | None = None
            try:
                sock = _connect_ws(self.url)
                with self.lock:
                    self._sock = sock
                    self.connected = True
                    self.last_error = None
                    self.last_addr = self.url
                if self.topics:
                    subscribe = json.dumps({"action": "subscribe", "topics": self.topics}).encode("utf-8")
                    _send_ws_frame(sock, 0x1, subscribe, masked=True)
                while not self._stop.is_set() and self._receiving.is_set():
                    try:
                        opcode, payload = _recv_ws_frame(sock)
                    except socket.timeout:
                        continue
                    if opcode == 0x8:
                        break
                    if opcode == 0x9:
                        _send_ws_frame(sock, 0xA, payload, masked=True)
                        continue
                    if opcode not in (0x1, 0x2):
                        continue
                    control_message = self._handle_control_message(payload)
                    if control_message == "ignore":
                        continue
                    with self.lock:
                        self.packet_count += 1
                        index = self.packet_count - 1
                    sample = parse_live_packet(payload, index=index)
                    with self.lock:
                        if sample is None:
                            self.drop_count += 1
                            self.last_error = f"无法解析 WebSocket 数据: {payload[:120]!r}"
                            continue
                        sample["source_addr"] = self.url
                        sample["transport"] = "websocket"
                        self.samples.append(sample)
                        self.last_error = None
            except Exception as exc:
                with self.lock:
                    self.connected = False
                    if self._receiving.is_set() and not self._stop.is_set():
                        self.reconnect_count += 1
                        self.last_error = str(exc)
            finally:
                with self.lock:
                    if self._sock is sock:
                        self._sock = None
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                with self.lock:
                    self.connected = False
            if not self._stop.wait(self.reconnect_delay):
                continue

    def _handle_control_message(self, payload: bytes) -> str | None:
        try:
            obj = json.loads(payload.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        if obj.get("type") == "subscription":
            with self.lock:
                self.last_error = None
            return "ignore"
        if obj.get("type") == "error":
            with self.lock:
                self.last_error = str(obj.get("message") or obj)
            return "ignore"
        return None

    def snapshot(self, max_points: int | None = None) -> dict[str, Any]:
        with self.lock:
            samples = list(self.samples)
            packet_count = self.packet_count
            drop_count = self.drop_count
            reconnect_count = self.reconnect_count
            connected = self.connected
            last_error = self.last_error
            last_addr = self.last_addr
            receiving = self.is_receiving()
        if max_points is not None and max_points > 0 and len(samples) > max_points:
            last = len(samples) - 1
            indices = sorted({round(index * last / (max_points - 1)) for index in range(max_points)})
            samples = [samples[index] for index in indices]
        first_epoch = next((sample["epoch"] for sample in samples if sample["epoch"] is not None), None)
        for fallback_t, sample in enumerate(samples):
            sample["index"] = fallback_t
            sample["t"] = fallback_t if first_epoch is None or sample["epoch"] is None else sample["epoch"] - first_epoch
        return {
            "log_file": self.url,
            "source": "live_websocket",
            "receiving": receiving,
            "connected": connected,
            "last_addr": last_addr,
            "count": len(samples),
            "packet_count": packet_count,
            "drop_count": drop_count,
            "reconnect_count": reconnect_count,
            "last_error": last_error,
            "kinds": {
                kind: sum(1 for sample in samples if sample["kind"] == kind)
                for kind in ("robot_target", "umi_raw", "robot_feedback")
            },
            "samples": samples,
        }


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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


class UmiVisualizerApp:
    def __init__(self, log_file: Path, receiver: Any | None = None) -> None:
        self.log_file = log_file.expanduser().resolve()
        self.receiver = receiver

    def data(self, max_points: int | None) -> dict[str, Any]:
        if self.receiver is not None:
            return self.receiver.snapshot(max_points=max_points)
        return parse_umi_log(self.log_file, max_points=max_points)

    def set_receiving(self, enabled: bool, max_points: int | None = None) -> dict[str, Any]:
        if self.receiver is None or not hasattr(self.receiver, "set_receiving"):
            raise RuntimeError("当前不是实时接收模式")
        self.receiver.set_receiving(enabled)
        return self.data(max_points=max_points)


class UmiVisualizerHandler(BaseHTTPRequestHandler):
    app: UmiVisualizerApp

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html_response(
                self,
                render_html_template("umi_state_visualizer.html", log_file=str(self.app.log_file)),
            )
            return
        if parsed.path == "/ws":
            self._serve_websocket(parsed.query)
            return
        if parsed.path == "/api/data":
            query = parse_qs(parsed.query)
            max_points = int(query.get("max_points", ["6000"])[0] or 6000)
            try:
                json_response(self, {"ok": True, "data": self.app.data(max_points)})
            except Exception as exc:
                json_response(self, {"ok": False, "error": str(exc), "data": {"samples": [], "count": 0}}, 200)
            return
        json_response(self, {"ok": False, "error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/receiving":
            self._handle_receiving_control()
            return
        json_response(self, {"ok": False, "error": "Not found"}, 404)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else {}

    def _handle_receiving_control(self) -> None:
        try:
            body = self._read_json_body()
            enabled = bool(body.get("receiving"))
            max_points = int(body.get("max_points") or 6000)
            data = self.app.set_receiving(enabled, max_points=max_points)
            json_response(self, {"ok": True, "data": data})
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc), "data": {"samples": [], "count": 0}}, 200)

    def _serve_websocket(self, query_text: str) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        upgrade = self.headers.get("Upgrade", "")
        if not key or upgrade.lower() != "websocket":
            json_response(self, {"ok": False, "error": "Expected WebSocket upgrade"}, 400)
            return

        query = parse_qs(query_text)
        max_points = int(query.get("max_points", ["6000"])[0] or 6000)
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", _websocket_accept_key(key))
        self.end_headers()
        self.close_connection = True

        last_marker: tuple[Any, ...] | None = None
        last_sent = 0.0
        while True:
            try:
                data = self.app.data(max_points=max_points)
                now = time.time()
                marker = (
                    data.get("source"),
                    data.get("count"),
                    data.get("packet_count"),
                    data.get("drop_count"),
                    data.get("reconnect_count"),
                    data.get("receiving"),
                    data.get("connected"),
                    data.get("last_error"),
                    data.get("last_addr"),
                )
                if marker != last_marker or now - last_sent > 2.0:
                    payload = json.dumps({"ok": True, "data": data}, ensure_ascii=False).encode("utf-8")
                    _send_ws_frame(self.connection, 0x1, payload, masked=False)
                    last_marker = marker
                    last_sent = now
                time.sleep(0.1)
            except Exception:
                break

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def port_available(host: str, port: int) -> bool:
    for probe in {host, "127.0.0.1", "0.0.0.0"}:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((probe, port))
            except OSError:
                return False
    return True


def make_server(host: str, preferred_port: int) -> ThreadingHTTPServer:
    port = preferred_port
    if preferred_port != 0:
        for candidate in range(preferred_port, preferred_port + 200):
            if port_available(host, candidate):
                port = candidate
                break
    server = ThreadingHTTPServer((host, port), UmiVisualizerHandler)
    if preferred_port != 0 and server.server_address[1] != preferred_port:
        print(f"Control port {preferred_port} is busy; using {server.server_address[1]}.")
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UMI arms state log visualizer.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_SAMPLE_LOG,
        help="log_umi_arms_state.py 生成的日志文件；默认使用示例日志",
    )
    parser.add_argument("--host", default="127.0.0.1", help="控制网页 host")
    parser.add_argument("--port", type=int, default=8060, help="控制网页端口")
    parser.add_argument("--live", action="store_true", help="实时监听 UMI 数据而不是读取日志文件")
    parser.add_argument(
        "--live-transport",
        choices=("websocket", "udp"),
        default="websocket",
        help="实时数据入口，默认从 UMI WebSocket 接收",
    )
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="UMI 上游 WebSocket 地址")
    parser.add_argument(
        "--ws-topic",
        action="append",
        default=None,
        help="UMI 上游 WebSocket 订阅 topic；可重复传入，默认 state",
    )
    parser.add_argument("--data-bind-host", default="0.0.0.0", help="UMI UDP 本地绑定地址")
    parser.add_argument("--data-port", type=int, default=8211, help="UMI UDP 本地监听端口")
    parser.add_argument(
        "--data-source-host",
        default="192.168.150.27",
        help="只接收该源 IP 的 UDP 数据；留空表示接收所有来源",
    )
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receiver = None
    if args.live:
        if args.live_transport == "websocket":
            receiver = WebSocketUmiReceiver(url=args.ws_url, topics=args.ws_topic or ["state"])
        else:
            receiver = LiveUmiReceiver(
                bind_host=args.data_bind_host,
                port=args.data_port,
                source_host=args.data_source_host or None,
            )
        receiver.start()
    app = UmiVisualizerApp(args.log_file, receiver=receiver)
    UmiVisualizerHandler.app = app
    server = make_server(args.host, args.port)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    if receiver is None:
        print(f"UMI log: {app.log_file}")
    elif isinstance(receiver, WebSocketUmiReceiver):
        print(f"UMI WebSocket: {receiver.url}, topics={','.join(receiver.topics) or '<none>'}")
    else:
        print(
            "UMI UDP: "
            f"bind={receiver.bind_host}:{receiver.port}, "
            f"source_filter={receiver.source_host or '*'}"
        )
    print(f"Control panel: {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
        if receiver is not None:
            receiver.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
