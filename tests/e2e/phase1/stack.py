# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Process management for the Phase 1 stack.

Topology (one node, two GPUs, aggregated, TP=1):

    frontend (dynamo.frontend --router-mode kv)
    worker[0]  GPU 0  DP=1  KVCR tier -> kvcr_service[0] (Guard 0)
    worker[1]  GPU 1  DP=1  KVCR tier -> kvcr_service[1] (Guard 0)

Every worker is its own process group so a test can kill one engine without
touching its KVCR service, which must outlive the engine for TC-04.
"""

from __future__ import annotations

import ctypes.util
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

from .client import Client
from .common import (
    BLOCK_SIZE,
    BlockedError,
    ValidationError,
    save_json,
    utc_now,
    worker_identity,
)

GUARD = "guard"
NON_GUARD = "nonguard"
DISABLED = "disabled"
MODES = (GUARD, NON_GUARD)
MIN_GUARD_KERNEL = (6, 5)
GUARD_PROMOTED = "KVCR_EVENT guard_promoted"
RECOVERY_DISABLED = "KVCR pool recovery disabled"


def kernel_release() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", platform.release())
    if match is None:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def check_port_available(port: int) -> None:
    """Reject active listeners; allow reuse after a prior service exited."""
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
    except OSError:
        pass
    else:
        connection.close()
        raise ValidationError(f"required TCP port {port} has an active listener")
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("0.0.0.0", port))
            listener.listen(1)
        except OSError as error:
            raise ValidationError(f"required TCP port {port} is unavailable: {error}")


@dataclass
class Settings:
    model: str
    output: Path
    mode: str = GUARD
    gpus: tuple[int, int] = (0, 1)
    http_port: int = 8000
    reference_http_port: int = 8001
    control_ports: tuple[int, int] = (17771, 17772)
    event_ports: tuple[int, int] = (20081, 20082)
    system_ports: tuple[int, int] = (9091, 9092)
    pool_gb: int = 2
    g2_slots: int = 256
    cpu_bytes_to_use: int = 2_000_000_000
    max_model_len: int = 32000
    gpu_memory_utilization: float = 0.8
    compatibility_digest: str = ""
    batch_invariant: bool = False
    readiness_timeout: float = 600
    request_timeout: float = 120
    extra_worker_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.compatibility_digest:
            self.compatibility_digest = (
                f"kvcr-phase1-{self.model}-{uuid.uuid4().hex[:8]}"
            )


class Managed:
    """A child process in its own session with a log file."""

    def __init__(self, name: str, command: list[str], env: dict[str, str], log: Path):
        self.name = name
        self.command = command
        self.log_path = log
        self.started_at = utc_now()
        self.stopped = False
        self._log = log.open("ab")
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except BaseException:
            self._log.close()
            raise

    @property
    def pid(self) -> int:
        return self.process.pid

    def alive(self) -> bool:
        return self.process.poll() is None

    def signal_group(self, number: int) -> bool:
        try:
            os.killpg(self.process.pid, number)
            return True
        except ProcessLookupError:
            return False

    def stop(self, grace: float = 20) -> dict[str, Any]:
        outcome: dict[str, Any] = {"name": self.name, "pid": self.process.pid}
        self.stopped = True
        self.signal_group(signal.SIGTERM)
        try:
            self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            outcome["sigkill"] = True
        # Descendants may survive their leader; sweep the whole group.
        self.signal_group(signal.SIGKILL)
        try:
            self.process.wait(timeout=10)
            outcome["exit_code"] = self.process.returncode
        except subprocess.TimeoutExpired:
            outcome["error"] = "process did not exit"
        self._log.close()
        return outcome

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace")


class Stack:
    def __init__(self, settings: Settings, evidence: Path) -> None:
        self.settings = settings
        self.evidence = evidence
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.children: list[Managed] = []
        self.discovery: tempfile.TemporaryDirectory | None = None
        self.services: dict[int, Managed] = {}
        self.workers: dict[int, Managed] = {}
        self.worker_ids: dict[int, str] = {}
        self.retired_ids: set[str] = set()
        self.frontend: Managed | None = None
        self.kvcr_run_dir = Path(tempfile.mkdtemp(prefix="phase1-run-"))
        self.pool_dir = Path("/dev/shm/phase1-pool-" + uuid.uuid4().hex[:8])
        self.deadline_check: Callable[[], None] = lambda: None

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> dict[str, Any]:
        import torch

        shm = os.statvfs("/dev/shm")
        data: dict[str, Any] = {
            "kernel": platform.release(),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "shm_available_bytes": shm.f_bavail * shm.f_frsize,
            "libatomic": ctypes.util.find_library("atomic"),
            "mode": self.settings.mode,
            "devices": [],
        }
        save_json(self.evidence / "preflight.json", data)
        if not data["cuda_available"] or data["device_count"] < 2:
            raise BlockedError("Phase 1 needs two usable CUDA GPUs on one node")
        for index in self.settings.gpus:
            value = torch.ones(16, device=f"cuda:{index}").sum().item()
            data["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "probe": value,
                }
            )
            if value != 16:
                raise ValidationError(f"CUDA compute probe failed on GPU {index}")
        if self.settings.mode == GUARD:
            if kernel_release() < MIN_GUARD_KERNEL:
                raise BlockedError(
                    "Guard mode needs Linux "
                    f"{MIN_GUARD_KERNEL[0]}.{MIN_GUARD_KERNEL[1]}+ "
                    f"(SO_PEERPIDFD); this node runs {platform.release()}"
                )
            if data["libatomic"] is None:
                raise BlockedError("Guard mode needs the system libatomic runtime")
            needed = 2 * (self.settings.pool_gb * 1024**3 + 128 * 1024**2)
            if data["shm_available_bytes"] < needed:
                raise BlockedError(
                    f"/dev/shm has {data['shm_available_bytes']} bytes free; two Guard "
                    f"pools need at least {needed}"
                )
        for port in (
            self.settings.http_port,
            self.settings.reference_http_port,
            *self.settings.control_ports,
            *self.settings.event_ports,
            *self.settings.system_ports,
        ):
            check_port_available(port)
        save_json(self.evidence / "preflight.json", data)
        return data

    # -- environment -------------------------------------------------------

    def base_env(self) -> dict[str, str]:
        if self.discovery is None:
            self.discovery = tempfile.TemporaryDirectory(prefix="phase1-disc-")
        env = os.environ.copy()
        env.pop("NATS_SERVER", None)
        env.update(
            DYN_REQUEST_PLANE="tcp",
            DYN_EVENT_PLANE="zmq",
            DYN_DISCOVERY_BACKEND="file",
            DYN_FILE_KV=self.discovery.name,
            PYTHONHASHSEED="0",
            VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES="1",
        )
        if self.settings.batch_invariant:
            env["VLLM_BATCH_INVARIANT"] = "1"
        return env

    def reset_discovery(self) -> None:
        if self.discovery is not None:
            self.discovery.cleanup()
            self.discovery = None

    # -- configuration -----------------------------------------------------

    def tier_config(self, index: int) -> dict[str, Any]:
        tier: dict[str, Any] = {
            "type": "kvcr",
            "router_capabilities": ["router_hint"],
            "control_host": "0.0.0.0",
            "control_ports": [self.settings.control_ports[index]],
            "control_advertise_host": "127.0.0.1",
            "operation_timeout_ms": 5000,
            "abandon_timeout_ms": 10000,
            "enable_telemetry": True,
        }
        if self.settings.mode == GUARD:
            tier["kvcr_service_socket_path"] = str(self.service_socket(index))
            tier["compatibility_digest"] = self.settings.compatibility_digest
        else:
            tier["secondary_g2_slots"] = self.settings.g2_slots
        return tier

    def transfer_config(self, index: int) -> dict[str, Any]:
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": self.settings.cpu_bytes_to_use,
                "self_describing_kv_events": True,
                "secondary_tiers": [self.tier_config(index)],
            },
        }

    def events_config(self, index: int) -> dict[str, Any]:
        return {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{self.settings.event_ports[index]}",
            "enable_kv_cache_events": True,
        }

    def worker_command(self, index: int, *, kvcr: bool) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "dynamo.vllm",
            "--model",
            self.settings.model,
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "1",
            "--block-size",
            str(BLOCK_SIZE),
            "--max-model-len",
            str(self.settings.max_model_len),
            "--enable-prefix-caching",
            "--disable-hybrid-kv-cache-manager",
            "--enforce-eager",
            "--gpu-memory-utilization",
            str(self.settings.gpu_memory_utilization),
            *self.settings.extra_worker_args,
        ]
        if kvcr:
            command += [
                "--kv-transfer-config",
                json.dumps(self.transfer_config(index)),
                "--kv-events-config",
                json.dumps(self.events_config(index)),
            ]
        return command

    def service_socket(self, index: int) -> Path:
        return self.kvcr_run_dir / f"memory-{index}.sock"

    # -- launching ---------------------------------------------------------

    def _spawn(self, name: str, command: list[str], env: dict[str, str]) -> Managed:
        log = self.evidence / f"{name}.log"
        child = Managed(name, command, env, log)
        self.children.append(child)
        save_json(
            self.evidence / f"{name}.launch.json",
            {
                "command": command,
                "pid": child.pid,
                "started_at": child.started_at,
                "environment_overrides": {
                    key: env[key]
                    for key in sorted(env)
                    if key.startswith(
                        ("DYN_", "VLLM_", "CUDA_", "KVCR_", "PYTHONHASHSEED")
                    )
                },
            },
        )
        return child

    def start_frontend(self, port: int, name: str = "frontend") -> Managed:
        command = [
            sys.executable,
            "-m",
            "dynamo.frontend",
            "--router-mode",
            "kv",
            "--http-port",
            str(port),
        ]
        self.frontend = self._spawn(name, command, self.base_env())
        return self.frontend

    def start_service(self, index: int) -> Managed:
        if self.settings.mode != GUARD:
            raise ValidationError("KVCR services are only launched in Guard mode")
        pool_dir = self.pool_dir / str(index)
        pool_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "kvcr.kvcr_service",
            "--socket-path",
            str(self.service_socket(index)),
            "--pool-dir",
            str(pool_dir),
            "--guard-count",
            "1",
            "--pool-sizes-gb",
            str(self.settings.pool_gb),
            "--compatibility-digest",
            self.settings.compatibility_digest,
            "--log-level",
            "DEBUG",
        ]
        service = self._spawn(f"kvcr-service-{index}", command, self.base_env())
        self.services[index] = service
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if not service.alive():
                raise ValidationError(
                    f"kvcr_service {index} exited early: {service.process.returncode}"
                )
            try:
                probe = socket.socket(socket.AF_UNIX)
                probe.settimeout(1)
                probe.connect(str(self.service_socket(index)))
                probe.close()
                return service
            except OSError:
                time.sleep(0.5)
        raise ValidationError(f"kvcr_service {index} socket did not become ready")

    def start_worker(self, index: int, *, kvcr: bool = True, tag: str = "") -> Managed:
        env = self.base_env()
        env["CUDA_VISIBLE_DEVICES"] = str(self.settings.gpus[index])
        env["VLLM_LOGGING_LEVEL"] = "DEBUG"
        env["DYN_SYSTEM_PORT"] = str(self.settings.system_ports[index])
        name = f"worker-{index}{tag}"
        save_json(
            self.evidence / f"{name}.kv-transfer-config.json",
            self.transfer_config(index) if kvcr else None,
        )
        worker = self._spawn(name, self.worker_command(index, kvcr=kvcr), env)
        self.workers[index] = worker
        previous = self.worker_ids.pop(index, None)
        if previous is not None:
            self.retired_ids.add(previous)
        return worker

    # -- readiness ---------------------------------------------------------

    def check_children(self, *names: str) -> None:
        for child in self.children:
            if names and child.name not in names:
                continue
            if child in self.workers.values() or child is self.frontend:
                if not child.alive():
                    raise ValidationError(
                        f"{child.name} exited early with status "
                        f"{child.process.returncode}"
                    )
            elif child in self.services.values() and not child.alive():
                raise ValidationError(
                    f"{child.name} exited with status {child.process.returncode}"
                )

    def wait_model(self, client: Client) -> dict[str, Any]:
        deadline = time.monotonic() + self.settings.readiness_timeout
        last = "model not registered"
        while time.monotonic() < deadline:
            self.deadline_check()
            self.check_children()
            try:
                data = client.models(timeout=5)
            except Exception as error:  # noqa: BLE001 - readiness polling
                last = str(error)
            else:
                if any(
                    m.get("id") == self.settings.model for m in data.get("data", [])
                ):
                    return data
            time.sleep(2)
        raise ValidationError(f"model readiness timed out: {last}")

    def discovery_instance_ids(self) -> set[str]:
        """Instance IDs registered in the file discovery store (decimal strings).

        Dynamo's file store names model/instance records by hex instance ID, so
        the directory itself lists every registered worker.
        """
        if self.discovery is None:
            return set()
        found: set[str] = set()
        root = Path(self.discovery.name)
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            key = unquote(path.relative_to(root).as_posix())
            parts = key.lower().split("/")
            if "instances" not in parts and "models" not in parts:
                continue
            if re.fullmatch(r"[0-9a-f]{1,16}", parts[-1]):
                found.add(str(int(parts[-1], 16)))
        return found

    def discover_worker(self, client: Client, index: int) -> str:
        """Learn the instance ID of worker[index].

        Workers start one at a time, so the first unknown decode_worker_id
        belongs to the worker most recently launched. Unhinted probes let the
        router choose; when it keeps choosing a known worker, candidates from
        the discovery store are confirmed with a pinned probe instead.
        """
        known = {wid for i, wid in self.worker_ids.items() if i != index}
        known |= self.retired_ids
        errors = 0
        attempt = 0
        deadline = time.monotonic() + self.settings.readiness_timeout
        while time.monotonic() < deadline:
            attempt += 1
            self.deadline_check()
            self.check_children()
            candidates = [
                wid for wid in self.discovery_instance_ids() if wid not in known
            ]
            pinned = candidates[0] if attempt % 2 and len(candidates) == 1 else None
            probe = client.completion(
                f"probe-w{index}-{attempt:02d}",
                f"identity probe {uuid.uuid4().hex}",
                worker=pinned,
                max_tokens=1,
                extra_fields=["worker_id"],
            )
            if probe.ok and probe.body is not None:
                worker, _ = worker_identity(probe.body)
                if worker is not None and worker not in known:
                    self.worker_ids[index] = worker
                    return worker
            else:
                errors += 1
            time.sleep(3)
        raise ValidationError(
            f"worker {index} was never selected by the router in {attempt} probes "
            f"({errors} probe errors; discovery candidates "
            f"{sorted(self.discovery_instance_ids() - known)})"
        )

    # -- failure injection -------------------------------------------------

    def kill_worker(self, index: int) -> dict[str, Any]:
        worker = self.workers.pop(index)
        killed_at = utc_now()
        worker.stopped = True
        worker.signal_group(signal.SIGKILL)
        try:
            worker.process.wait(timeout=30)
        except subprocess.TimeoutExpired as error:
            raise ValidationError(f"worker {index} survived SIGKILL") from error
        retired = self.worker_ids.pop(index, None)
        if retired is not None:
            self.retired_ids.add(retired)
        return {
            "worker": worker.name,
            "pid": worker.pid,
            "killed_at": killed_at,
            "instance_id": retired,
        }

    def wait_log(
        self, child: Managed, needle: str, timeout: float, count: int = 1
    ) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.deadline_check()
            text = child.log_text()
            if text.count(needle) >= count:
                return text
            if not child.alive():
                raise ValidationError(
                    f"{child.name} exited while waiting for {needle!r}"
                )
            time.sleep(1)
        raise ValidationError(f"{child.name} did not log {needle!r} within {timeout}s")

    # -- teardown ----------------------------------------------------------

    def stop(self, *children: Managed) -> list[dict[str, Any]]:
        outcomes = []
        targets = list(children) or list(reversed(self.children))
        for child in targets:
            if not child.stopped:
                outcomes.append(child.stop())
        return outcomes

    def cleanup(self) -> dict[str, Any]:
        outcomes = self.stop()
        self.reset_discovery()
        removed = []
        for path in (self.kvcr_run_dir, self.pool_dir):
            if path.exists():
                for item in sorted(path.rglob("*"), reverse=True):
                    try:
                        item.unlink() if not item.is_dir() else item.rmdir()
                    except OSError:
                        pass
                try:
                    path.rmdir()
                    removed.append(str(path))
                except OSError:
                    pass
        report = {"owned_processes": outcomes, "removed_paths": removed}
        save_json(self.evidence / "cleanup.json", report)
        if any("error" in outcome for outcome in outcomes):
            raise ValidationError("owned process cleanup did not finish")
        return report
