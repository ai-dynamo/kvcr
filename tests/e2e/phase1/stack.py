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
import urllib.error
import urllib.request
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
STATE_AGENT_READY = "KV state-agent host started"
CACHE_OWNER_DOMAIN = (
    "01010101010101010101010101010101:02020202020202020202020202020202/0303030303030303"
)
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
    # "agg": two decode workers, worker 0 KVCR source, worker 1 target.
    # "disagg": 2P+1D on three GPUs: workers 0/1 are prefill workers (KVCR
    # source/target, the P->D hop uses NixlConnector), worker 2 is the single
    # decode worker. Requests pin the prefill worker; the decode is shared.
    topology: str = "agg"
    gpus: tuple[int, ...] = (0, 1)
    http_port: int = 8000
    reference_http_port: int = 8001
    control_ports: tuple[int, ...] = (17771, 17772, 17773)
    event_ports: tuple[int, ...] = (20081, 20082, 20083)
    system_ports: tuple[int, ...] = (9091, 9092, 9093)
    nixl_side_channel_ports: tuple[int, ...] = (5601, 5602, 5603)
    pool_gb: int = 2
    g2_slots: int = 256
    cpu_bytes_to_use: int = 2_000_000_000
    max_model_len: int = 32000
    gpu_memory_utilization: float = 0.8
    # Two-node run: worker index -> node-agent URL on the other node
    # (tests/e2e/phase1/node_agent.py); that node also hosts the worker's
    # kvcr_service sidecar. Needs etcd_endpoints: Dynamo's file discovery
    # watches with inotify, which does not cross nodes.
    remote_workers: dict[int, str] | None = None
    etcd_endpoints: str | None = None
    compatibility_digest: str = ""
    batch_invariant: bool = False
    state_agent: bool = True
    # KVCR's quick-start disables vLLM's hybrid KV cache manager. Models with
    # non-attention layers (Qwen3.5 GDN, Mamba hybrids) cannot be unified into
    # one KV group, so they need the manager kept on.
    disable_hybrid_kv_cache_manager: bool = True
    # Eager mode keeps the stack simple, but GDN/Mamba layers without CUDA
    # graphs decode very slowly (Qwen3.5 reference phase took ~40 min).
    enforce_eager: bool = True
    # Sidecar topology: kvcr_service instances run in their own containers and
    # share this directory (sockets + logs) with the vLLM container. None means
    # the harness launches the services itself in-process.
    external_services: Path | None = None
    readiness_timeout: float = 600
    request_timeout: float = 120
    extra_worker_args: list[str] = field(default_factory=list)
    # Extra environment for the Dynamo frontend only (e.g. DYN_LOG for router
    # debugging).
    frontend_env: dict[str, str] = field(default_factory=dict)

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


class _RemoteProcess:
    """The subset of subprocess.Popen the harness touches, backed by a node agent."""

    def __init__(self, owner: "RemoteManaged") -> None:
        self._owner = owner
        self.pid = owner.pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        status = self._owner.agent("GET", f"/status/{self._owner.name}")
        self.returncode = status.get("returncode")
        return None if status.get("alive") else self.returncode

    def wait(self, timeout: float = 30) -> int | None:
        outcome = self._owner.agent(
            "POST", f"/wait/{self._owner.name}", {"timeout": timeout}
        )
        if not outcome.get("exited"):
            raise subprocess.TimeoutExpired(self._owner.command, timeout)
        self.returncode = outcome.get("returncode")
        return self.returncode


class RemoteManaged:
    """A child process on another node, driven through node_agent.py.

    Same surface as Managed; the log lives on a volume both nodes mount, so
    the checks read it like a local file.
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        env: dict[str, str],
        log: Path,
        agent_url: str,
    ) -> None:
        self.name = name
        self.command = command
        self.log_path = log
        self.agent_url = agent_url.rstrip("/")
        self.started_at = utc_now()
        self.stopped = False
        reply = self.agent(
            "POST",
            "/spawn",
            {"name": name, "command": command, "env": env, "log": str(log)},
        )
        self.pid = int(reply["pid"])
        self.process = _RemoteProcess(self)

    def agent(
        self, method: str, route: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return node_agent_call(self.agent_url, method, route, body)

    def alive(self) -> bool:
        return self.process.poll() is None

    def signal_group(self, number: int) -> bool:
        return bool(
            self.agent("POST", f"/signal/{self.name}", {"signal": int(number)}).get(
                "ok"
            )
        )

    def stop(self, grace: float = 20) -> dict[str, Any]:
        outcome: dict[str, Any] = {
            "name": self.name,
            "pid": self.pid,
            "remote": self.agent_url,
        }
        self.stopped = True
        self.signal_group(signal.SIGTERM)
        try:
            self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            outcome["sigkill"] = True
        self.signal_group(signal.SIGKILL)
        try:
            outcome["exit_code"] = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            outcome["error"] = "process did not exit"
        return outcome

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            pass
        try:
            return node_agent_text(self.agent_url, f"/log/{self.name}")
        except (OSError, ValueError):
            return ""


def node_agent_call(
    base: str,
    method: str,
    route: str,
    body: dict[str, Any] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise ValidationError(
            f"node agent {base}{route} -> {error.code}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise ValidationError(
            f"node agent {base}{route} unreachable: {error.reason}"
        ) from error


def node_agent_text(base: str, route: str, timeout: float = 60) -> str:
    with urllib.request.urlopen(base.rstrip("/") + route, timeout=timeout) as response:
        return response.read().decode(errors="replace")


class ExternalService:
    """A kvcr_service that lives in another container; observed, not owned.

    Liveness is a connect() to its Unix socket; its log is a file the sidecar
    writes into the shared directory. stop() is a no-op: the sidecar's lifetime
    belongs to the orchestrator, which is exactly what TC-04 relies on.
    """

    def __init__(
        self, name: str, socket_path: Path, log_path: Path, agent_url: str | None = None
    ) -> None:
        self.name = name
        self.socket_path = socket_path
        self.log_path = log_path
        self.agent_url = agent_url  # socket lives on another node: probe via its agent
        self.stopped = False
        self.started_at = utc_now()

    def alive(self) -> bool:
        if self.agent_url is not None:
            try:
                return bool(
                    node_agent_call(
                        self.agent_url, "GET", f"/socket?path={self.socket_path}"
                    ).get("alive")
                )
            except ValidationError:
                return False
        try:
            probe = socket.socket(socket.AF_UNIX)
            probe.settimeout(1)
            probe.connect(str(self.socket_path))
            probe.close()
            return True
        except OSError:
            return False

    def log_text(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def stop(self, grace: float = 0) -> dict[str, Any]:
        return {"name": self.name, "external": True}


class Stack:
    def __init__(self, settings: Settings, evidence: Path) -> None:
        self.settings = settings
        self.evidence = evidence
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.children: list[Managed] = []
        self.discovery: tempfile.TemporaryDirectory | None = None
        self.services: dict[int, Managed] = {}
        self.state_agent: Managed | None = None
        self.workers: dict[int, Managed] = {}
        self.worker_ids: dict[int, str] = {}
        self.retired_ids: set[str] = set()
        self.frontend: Managed | None = None
        self.kvcr_run_dir = Path(tempfile.mkdtemp(prefix="phase1-run-"))
        self.pool_dir = Path("/dev/shm/phase1-pool-" + uuid.uuid4().hex[:8])
        self.deadline_check: Callable[[], None] = lambda: None
        self._remote_info: dict[int, dict[str, Any]] = {}

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
        needed = max(self.settings.gpus) + 1
        if not data["cuda_available"] or data["device_count"] < needed:
            raise BlockedError(f"Phase 1 needs {needed} usable CUDA GPU(s) on one node")
        for index in sorted(set(self.settings.gpus)):
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
            for index, agent in sorted((self.settings.remote_workers or {}).items()):
                # The other node's pod may still be pulling its image.
                deadline = time.monotonic() + self.settings.readiness_timeout
                while True:
                    try:
                        info = node_agent_call(agent, "GET", "/info", timeout=10)
                        break
                    except ValidationError as error:
                        if time.monotonic() > deadline:
                            raise BlockedError(
                                f"node agent for worker {index} never answered: {error}"
                            ) from error
                        time.sleep(5)
                self._remote_info[index] = info
                data.setdefault("remote_nodes", {})[str(index)] = info
                found = re.match(r"(\d+)\.(\d+)", str(info.get("kernel", "")))
                if found is None or (int(found[1]), int(found[2])) < MIN_GUARD_KERNEL:
                    raise BlockedError(
                        f"remote node for worker {index} ({info.get('hostname')}) runs "
                        f"{info.get('kernel')}; Guard needs Linux "
                        f"{MIN_GUARD_KERNEL[0]}.{MIN_GUARD_KERNEL[1]}+"
                    )
            if data["libatomic"] is None:
                raise BlockedError("Guard mode needs the system libatomic runtime")
            needed = 2 * (self.settings.pool_gb * 1024**3 + 128 * 1024**2)
            if (
                self.settings.external_services is None
                and data["shm_available_bytes"] < needed
            ):
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
            PYTHONHASHSEED="0",
            VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES="1",
        )
        if self.settings.etcd_endpoints:
            env["DYN_DISCOVERY_BACKEND"] = "etcd"
            env["ETCD_ENDPOINTS"] = self.settings.etcd_endpoints
        else:
            env["DYN_DISCOVERY_BACKEND"] = "file"
            env["DYN_FILE_KV"] = self.discovery.name
        if self.settings.remote_workers:
            # Peers on other nodes must reach this process: advertise a real
            # address (Dynamo's event plane defaults to 127.0.0.1).
            env["DYN_EVENT_PLANE_HOST"] = self.advertise_host()
            env["DYN_TCP_RPC_HOST"] = self.advertise_host()
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
            "control_advertise_host": self.worker_host(index),
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

    @staticmethod
    def cache_owner_id(slot: int) -> str:
        """Stable per-slot CacheOwnerId; a replacement worker reuses its slot."""
        return f"{CACHE_OWNER_DOMAIN}/{slot:032x}"

    @staticmethod
    def advertise_host() -> str:
        """A non-loopback address of this host for the state agent's raw ZMQ link."""
        host = socket.gethostbyname(socket.gethostname())
        if host.startswith("127."):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("10.255.255.255", 1))
                host = probe.getsockname()[0]
        return host

    def remote_agent(self, index: int) -> str | None:
        return (self.settings.remote_workers or {}).get(index)

    def worker_host(self, index: int) -> str:
        """Address peers use for worker[index]'s KVCR control endpoint."""
        if not self.settings.remote_workers:
            return "127.0.0.1"
        agent = self.remote_agent(index)
        if agent is None:
            return self.advertise_host()
        if index not in self._remote_info:
            self._remote_info[index] = node_agent_call(agent, "GET", "/info")
        return str(self._remote_info[index]["ip"])

    def uses_state_agent(self) -> bool:
        return self.settings.mode == GUARD and self.settings.state_agent

    def transfer_config(self, index: int) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": self.settings.cpu_bytes_to_use,
            "self_describing_kv_events": True,
            "secondary_tiers": [self.tier_config(index)],
        }
        if self.uses_state_agent():
            extra["dynamo_state_agent"] = {
                "raw_advertise_host": self.advertise_host(),
                "cache_owner_ids": {"0": self.cache_owner_id(index)},
            }
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }

    # -- topology helpers --------------------------------------------------

    @property
    def disagg(self) -> bool:
        return self.settings.topology == "disagg"

    @property
    def decode_index(self) -> int | None:
        return 2 if self.disagg else None

    @property
    def served_role(self) -> str:
        """Which worker id identifies the KVCR-relevant worker of a response."""
        return "prefill" if self.disagg else "decode"

    def identity(self, body: dict[str, Any]) -> str | None:
        return worker_identity(body, self.served_role)[0]

    def served(self, stream: Any) -> str | None:
        return stream.prefill_worker if self.disagg else stream.worker

    def pin(self, index: int) -> dict[str, str | None]:
        """Request kwargs that route a request through worker[index]."""
        if not self.disagg:
            return {"worker": self.worker_ids.get(index)}
        return {
            "prefill": self.worker_ids.get(index),
            "worker": self.worker_ids.get(2),
        }

    def disaggregation_mode(self, index: int) -> str | None:
        if not self.disagg:
            return None
        return "decode" if index == 2 else "prefill"

    def kv_transfer_config(self, index: int, *, kvcr: bool) -> dict[str, Any] | None:
        mode = self.disaggregation_mode(index)
        if mode is None:
            return self.transfer_config(index) if kvcr else None
        if mode == "decode":
            return {"kv_connector": "NixlConnector", "kv_role": "kv_consumer"}
        nixl = {"kv_connector": "NixlConnector", "kv_role": "kv_producer"}
        if not kvcr:
            return nixl
        kvcr_cfg = self.transfer_config(index)
        extra: dict[str, Any] = {"connectors": [kvcr_cfg, nixl]}
        # Dynamo (ea23e51) reads the KVCR tier list, the state-agent settings
        # and the self-describing-events flag from the TOP-LEVEL
        # kv_connector_extra_config to advertise router-hint capability and
        # attach the KV state agent; nested inside MultiConnector they are
        # invisible, so no hints reach a prefill worker. Mirror them at the top
        # level (vLLM's MultiConnector ignores keys other than "connectors").
        for key in (
            "secondary_tiers",
            "dynamo_state_agent",
            "self_describing_kv_events",
        ):
            if key in kvcr_cfg["kv_connector_extra_config"]:
                extra[key] = kvcr_cfg["kv_connector_extra_config"][key]
        return {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
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
            *(
                ["--disable-hybrid-kv-cache-manager"]
                if self.settings.disable_hybrid_kv_cache_manager
                else []
            ),
            *(["--enforce-eager"] if self.settings.enforce_eager else []),
            "--gpu-memory-utilization",
            str(self.settings.gpu_memory_utilization),
            *self.settings.extra_worker_args,
        ]
        mode = self.disaggregation_mode(index)
        if mode is not None:
            command += ["--disaggregation-mode", mode]
        transfer = self.kv_transfer_config(index, kvcr=kvcr)
        if transfer is not None:
            command += ["--kv-transfer-config", json.dumps(transfer)]
        if kvcr or (mode is not None):
            command += ["--kv-events-config", json.dumps(self.events_config(index))]
        return command

    def service_socket(self, index: int) -> Path:
        if self.settings.external_services is not None:
            return self.settings.external_services / f"memory-{index}.sock"
        return self.kvcr_run_dir / f"memory-{index}.sock"

    def attach_service(self, index: int) -> ExternalService:
        """Sidecar topology: wait for the shared socket instead of launching."""
        assert self.settings.external_services is not None
        agent = self.remote_agent(index)
        service = ExternalService(
            f"kvcr-service-{index}",
            self.service_socket(index),
            # A remote sidecar writes its log into the shared evidence tree.
            (self.evidence if agent else self.settings.external_services)
            / f"kvcr-service-{index}.log",
            agent_url=agent,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if service.alive():
                self.services[index] = service  # type: ignore[assignment]
                save_json(
                    self.evidence / f"{service.name}.launch.json",
                    {
                        "external": True,
                        "socket": str(service.socket_path),
                        "log": str(service.log_path),
                        "attached_at": utc_now(),
                    },
                )
                return service
            time.sleep(1)
        raise ValidationError(
            f"external kvcr_service {index} socket {service.socket_path} not reachable"
        )

    # -- launching ---------------------------------------------------------

    def _spawn(
        self,
        name: str,
        command: list[str],
        env: dict[str, str],
        agent_url: str | None = None,
    ) -> Managed:
        log = self.evidence / f"{name}.log"
        if agent_url is not None:
            # Only pass what differs from the agent's own environment: the
            # other node keeps its HF cache path, PATH and CUDA setup.
            overrides = {k: v for k, v in env.items() if os.environ.get(k) != v}
            child: Managed = RemoteManaged(name, command, overrides, log, agent_url)  # type: ignore[assignment]
        else:
            child = Managed(name, command, env, log)
        self.children.append(child)
        save_json(
            self.evidence / f"{name}.launch.json",
            {
                "command": command,
                "pid": child.pid,
                "remote": agent_url,
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
        env = self.base_env()
        env.update(self.settings.frontend_env)
        self.frontend = self._spawn(name, command, env)
        return self.frontend

    def start_state_agent(self) -> Managed:
        """Run Dynamo's KV state-agent host so a dead engine's cache stays routable.

        Without it the router drops a vanished worker's blocks and never hints
        at its Guard. One host serves both worker slots.
        """
        command = [
            sys.executable,
            "-m",
            "dynamo.kv_state_agent",
            "--max-slots",
            "2",
        ]
        env = self.base_env()
        env["DYN_SYSTEM_PORT"] = str(max(self.settings.system_ports) + 1)
        agent = self._spawn("kv-state-agent", command, env)
        self.state_agent = agent
        self.wait_log(agent, STATE_AGENT_READY, 120)
        return agent

    def start_service(self, index: int) -> Managed:
        if self.settings.mode != GUARD:
            raise ValidationError("KVCR services are only launched in Guard mode")
        if self.settings.external_services is not None:
            return self.attach_service(index)  # type: ignore[return-value]
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
        if self.disagg:
            # Three NIXL side channels on one host; Dynamo only sets the host
            # when it recognises a top-level NixlConnector.
            env.setdefault("VLLM_NIXL_SIDE_CHANNEL_HOST", self.worker_host(index))
            env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(
                self.settings.nixl_side_channel_ports[index]
            )
        name = f"worker-{index}{tag}"
        save_json(
            self.evidence / f"{name}.kv-transfer-config.json",
            self.transfer_config(index) if kvcr else None,
        )
        agent = self.remote_agent(index)
        if agent is not None:
            host = self.worker_host(index)
            env["DYN_EVENT_PLANE_HOST"] = host
            env["DYN_TCP_RPC_HOST"] = host
            worker = self._spawn(
                name, self.worker_command(index, kvcr=kvcr), env, agent_url=agent
            )
        else:
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
            elif (
                child in list(self.services.values()) or child is self.state_agent
            ) and not child.alive():
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
                last = (
                    f"model not listed yet: "
                    f"{[m.get('id') for m in data.get('data', [])]}"
                )
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
            if len(parts) < 2 or parts[-2] != "generate":
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
            role = "decode" if (not self.disagg or index == 2) else "prefill"
            pin: dict[str, str | None] = {"worker": pinned}
            if self.disagg and index != 2:
                # The prefill router rejects a request that pins only the decode
                # worker, so pin both or neither.
                pin = (
                    {"prefill": pinned, "worker": self.worker_ids.get(2)}
                    if pinned is not None
                    else {}
                )
            probe = client.completion(
                f"probe-w{index}-{attempt:02d}",
                f"identity probe {uuid.uuid4().hex}",
                max_tokens=1,
                extra_fields=["worker_id"],
                **pin,
            )
            if probe.ok and probe.body is not None:
                worker, _ = worker_identity(probe.body, role)
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
