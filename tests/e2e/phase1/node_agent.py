"""Minimal process host for the second node of a two-node Phase 1 run.

The harness on node A drives node B through this HTTP agent: spawn a worker
with a given command/env (stdout+stderr to a log file on the shared volume),
poll liveness, send signals, probe a Unix socket (the kvcr_service sidecar).
Stdlib only; no authentication - it is meant for a private pod network.

    python3 -m phase1.node_agent --port 9800 [--bind 0.0.0.0]

Endpoints (JSON bodies):
    GET  /info                      {"ip","hostname","kernel","pid"}
    POST /spawn                     {"name","command":[...],"env":{...},"log":"/path"} -> {"pid"}
    GET  /status/<name>             {"alive","returncode","pid"}
    POST /signal/<name>             {"signal": 9} -> {"ok"}
    POST /wait/<name>               {"timeout": 30} -> {"exited","returncode"}
    GET  /socket?path=/run/x.sock   {"alive"}
    GET  /log/<name>                text/plain (fallback when the log file is not shared)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_children: dict[str, tuple[subprocess.Popen, str]] = {}
_lock = threading.Lock()


def local_ip() -> str:
    """The address peers should use: the interface that routes outward."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        pass
    try:
        host = socket.gethostbyname(socket.gethostname())
    except OSError:
        host = "127.0.0.1"
    return host


class Handler(BaseHTTPRequestHandler):
    server_version = "kvcr-phase1-node-agent/1"

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _text(self, code: int, text: str) -> None:
        data = text.encode(errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _child(self, name: str) -> tuple[subprocess.Popen, str] | None:
        with _lock:
            return _children.get(name)

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        if "/status/" in (args[0] if args else ""):
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/info":
            self._json(200, {"ip": local_ip(), "hostname": socket.gethostname(),
                             "kernel": platform.release(), "pid": os.getpid(),
                             "children": sorted(_children)})
            return
        if url.path.startswith("/status/"):
            entry = self._child(url.path[len("/status/"):])
            if entry is None:
                self._json(404, {"error": "unknown child"})
                return
            proc, _ = entry
            code = proc.poll()
            self._json(200, {"alive": code is None, "returncode": code, "pid": proc.pid})
            return
        if url.path == "/socket":
            path = parse_qs(url.query).get("path", [""])[0]
            alive = False
            try:
                probe = socket.socket(socket.AF_UNIX)
                probe.settimeout(1)
                probe.connect(path)
                probe.close()
                alive = True
            except OSError:
                pass
            self._json(200, {"alive": alive, "path": path})
            return
        if url.path.startswith("/log/"):
            entry = self._child(url.path[len("/log/"):])
            if entry is None:
                self._json(404, {"error": "unknown child"})
                return
            try:
                with open(entry[1], errors="replace") as handle:
                    self._text(200, handle.read())
            except OSError as error:
                self._text(500, str(error))
            return
        self._json(404, {"error": "no such route"})

    def do_POST(self) -> None:
        url = urlparse(self.path)
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as error:
            self._json(400, {"error": f"bad json: {error}"})
            return
        if url.path == "/spawn":
            name, command, log = body.get("name"), body.get("command"), body.get("log")
            if not (isinstance(name, str) and isinstance(command, list) and isinstance(log, str)):
                self._json(400, {"error": "need name, command[], log"})
                return
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in (body.get("env") or {}).items()})
            with _lock:
                if name in _children and _children[name][0].poll() is None:
                    self._json(409, {"error": "child still running"})
                    return
                os.makedirs(os.path.dirname(log), exist_ok=True)
                handle = open(log, "ab")
                try:
                    proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                            env=env, start_new_session=True,
                                            cwd=body.get("cwd") or "/tmp")
                except OSError as error:
                    handle.close()
                    self._json(500, {"error": str(error)})
                    return
                _children[name] = (proc, log)
            self._json(200, {"pid": proc.pid, "log": log})
            return
        if url.path.startswith("/signal/"):
            entry = self._child(url.path[len("/signal/"):])
            if entry is None:
                self._json(404, {"error": "unknown child"})
                return
            number = int(body.get("signal", signal.SIGTERM))
            try:
                os.killpg(entry[0].pid, number)
                self._json(200, {"ok": True})
            except ProcessLookupError:
                self._json(200, {"ok": False})
            return
        if url.path.startswith("/wait/"):
            entry = self._child(url.path[len("/wait/"):])
            if entry is None:
                self._json(404, {"error": "unknown child"})
                return
            try:
                code = entry[0].wait(timeout=float(body.get("timeout", 30)))
                self._json(200, {"exited": True, "returncode": code})
            except subprocess.TimeoutExpired:
                self._json(200, {"exited": False, "returncode": None})
            return
        self._json(404, {"error": "no such route"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9800)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"node agent on {args.bind}:{args.port} ip={local_ip()} kernel={platform.release()}",
          flush=True)
    try:
        server.serve_forever()
    finally:
        with _lock:
            for proc, _ in _children.values():
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    main()
