# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Verify real vLLM/Dynamo KVCR peer transfer; never substitutes mocked GPU work.

Run inside the integration image on two visible GPUs. Supplying both --base-url
and --worker-log verifies an existing service without starting or stopping it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

BLOCK_SIZE = 64
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC = re.compile(
    r"(vllm:[^=]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?:, |$)"
)
SOURCE_BLOCKS = "vllm:kvcr_transfer_blocks:('source_write',)"
DESTINATION_BLOCKS = "vllm:kvcr_transfer_blocks:('remote_deliver',)"
SOURCE_BYTES = "vllm:kvcr_transfer_bytes:('source_write',)"
DESTINATION_BYTES = "vllm:kv_offload_tiering_read_bytes:('1:kvcr',)"
SOURCE_SUCCESS = "vllm:kvcr_duration_seconds:('source_write', 'success')_count"
TRANSFER_SUCCESS = "vllm:kvcr_duration_seconds:('transfer', 'success')_count"
DESTINATION_SUCCESS = "vllm:kvcr_duration_seconds:('remote_deliver', 'success')_count"
CASCADE_FAILURE = "vllm:kv_offload_tiering_cascade_job_failures:('1:kvcr',)"


class ValidationError(RuntimeError):
    """The observed deployment or evidence does not satisfy the test contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_junit(output: Path, result: dict[str, Any]) -> None:
    """Expose the complete transfer contract as one CI test, including cleanup."""
    passed = result.get("status") == "passed"
    started = datetime.fromisoformat(result["started_at"])
    finished = datetime.fromisoformat(result["finished_at"])
    elapsed = str(max(0.0, (finished - started).total_seconds()))
    suite = ET.Element(
        "testsuite",
        name="kvcr.vllm_e2e",
        tests="1",
        failures="0" if passed else "1",
        errors="0",
        skipped="0",
        time=elapsed,
    )
    case = ET.SubElement(
        suite,
        "testcase",
        classname="kvcr.vllm_e2e",
        name="rank0_seed_rank1_retrieval",
        time=elapsed,
    )
    if not passed:
        # Keep arbitrary logs and URLs out of the XML; detailed, sanitized
        # evidence is retained separately beside this report.
        failure = ET.SubElement(
            case,
            "failure",
            type="KVCRValidationFailure",
            message="KVCR end-to-end validation failed; inspect result.json",
        )
        failure.text = (
            "The complete contract includes environment readiness, rank identity, "
            "cache reuse, peer transfer counters, output consistency when enabled, "
            "and process cleanup. See result.json and telemetry.json."
        )
    temporary = output / "junit.xml.tmp"
    ET.ElementTree(suite).write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(output / "junit.xml")


def log_window_boundary(original: bytes) -> int:
    """Keep a final engine header if its metric line has not been written yet."""
    complete_end = original.rfind(b"\n") + 1
    cursor = complete_end
    while cursor:
        start = original.rfind(b"\n", 0, cursor - 1) + 1
        line = ANSI.sub("", original[start:cursor].decode(errors="replace")).strip()
        if line:
            return start if re.search(r"\bEngine (\d+):", line) else complete_end
        cursor = start
    return complete_end


def check_port_available(port: int) -> None:
    """Reject active TCP listeners while allowing reuse after a prior service."""
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
            raise ValidationError(
                f"required TCP port {port} is unavailable: {error}"
            ) from error


def parse_metrics(text: str, *, line_offset: int = 0) -> dict[str, Any]:
    """Parse complete physical log lines, retaining strict engine attribution.

    Metrics have tuple labels containing commas, so splitting on commas corrupts
    them. Each reset-interval sample must follow its own engine statistics line.
    The trailing incomplete line is deferred until a subsequent read.
    """
    snapshots: list[dict[str, Any]] = []
    errors: list[str] = []
    previous_line = ""
    previous_number = 0
    for local_number, raw in enumerate(text.split("\n")[:-1], 1):
        number = line_offset + local_number
        line = ANSI.sub("", raw).strip()
        if not line:
            continue
        if "KV Transfer metrics:" in line:
            engine_match = re.search(r"\bEngine (\d+):", previous_line)
            if engine_match is None:
                errors.append(f"line {number}: metrics lack adjacent engine context")
            else:
                stamp = line.split()[0]
                try:
                    datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                except ValueError:
                    errors.append(f"line {number}: missing ISO metric timestamp")
                else:
                    payload = line.split("KV Transfer metrics:", 1)[1].strip()
                    pairs = METRIC.findall(payload)
                    metrics = {
                        name: float(value)
                        if any(character in value for character in ".eE")
                        else int(value)
                        for name, value in pairs
                    }
                    if not metrics or len(metrics) != payload.count("="):
                        errors.append(f"line {number}: incomplete metric parsing")
                    else:
                        snapshots.append(
                            {
                                "engine": int(engine_match.group(1)),
                                "timestamp": stamp,
                                "line": number,
                                "engine_context_line": previous_number,
                                "metrics": metrics,
                            }
                        )
        previous_line, previous_number = line, number
    return {"snapshots": snapshots, "parse_errors": errors}


def assess_metrics(
    parsed: dict[str, Any],
    expected_blocks: int,
    *,
    retrieval_started_at: str,
    final: bool = False,
) -> dict[str, Any]:
    """Sum reset intervals; distinguish absent proof from explicit failures."""
    totals: dict[int, dict[str, int | float]] = defaultdict(dict)
    failures = list(parsed["parse_errors"])
    positive_failures = []
    cascades = []
    after_retrieval: set[int] = set()
    started = datetime.fromisoformat(retrieval_started_at.replace("Z", "+00:00"))
    for snapshot in parsed["snapshots"]:
        engine = snapshot["engine"]
        if engine not in (0, 1):
            failures.append(f"unexpected engine {engine} at line {snapshot['line']}")
        stamp = datetime.fromisoformat(snapshot["timestamp"].replace("Z", "+00:00"))
        if stamp >= started:
            after_retrieval.add(engine)
        for name, value in snapshot["metrics"].items():
            # A later empty report cannot make an older transfer evidence for
            # this retrieval. Retain older snapshots for diagnostics only.
            if stamp >= started:
                totals[engine][name] = totals[engine].get(name, 0) + value
            if (
                value > 0
                and name.startswith("vllm:kvcr_")
                and ("'failed'" in name or "'partial'" in name)
            ):
                positive_failures.append(
                    {
                        "engine": engine,
                        "metric": name,
                        "value": value,
                        "line": snapshot["line"],
                    }
                )
            if name == CASCADE_FAILURE and value > 0:
                cascades.append(
                    {"engine": engine, "value": value, "line": snapshot["line"]}
                )
    if positive_failures:
        failures.append("positive KVCR failed/partial metrics were observed")
    required = (
        (0, TRANSFER_SUCCESS),
        (0, SOURCE_SUCCESS),
        (1, DESTINATION_SUCCESS),
        (0, SOURCE_BLOCKS),
        (1, DESTINATION_BLOCKS),
        (0, SOURCE_BYTES),
        (1, DESTINATION_BYTES),
    )
    missing = [
        f"engine {engine}: {name}"
        for engine, name in required
        if name not in totals.get(engine, {})
    ]
    if not {0, 1}.issubset(after_retrieval):
        missing.append("post-retrieval interval samples for both engines")
    for engine, name in required:
        value = totals.get(engine, {}).get(name)
        if value is not None and value <= 0:
            message = f"engine {engine}: {name} is not positive ({value})"
            (failures if final or value < 0 else missing).append(message)
    source_blocks = totals.get(0, {}).get(SOURCE_BLOCKS)
    destination_blocks = totals.get(1, {}).get(DESTINATION_BLOCKS)
    source_bytes = totals.get(0, {}).get(SOURCE_BYTES)
    destination_bytes = totals.get(1, {}).get(DESTINATION_BYTES)
    for label, observed in (
        ("source", source_blocks),
        ("destination", destination_blocks),
    ):
        if observed is not None and observed != expected_blocks:
            message = f"{label} blocks {observed} != cached blocks {expected_blocks}"
            if final or observed > expected_blocks:
                failures.append(message)
            else:
                missing.append(message)
    if source_bytes is not None and destination_bytes is not None:
        if source_bytes != destination_bytes:
            message = (
                f"source bytes {source_bytes} != destination bytes {destination_bytes}"
            )
            (failures if final else missing).append(message)
    return {
        "status": "failed" if failures else "incomplete" if missing else "passed",
        "failures": failures,
        "missing": missing,
        "expected_blocks": expected_blocks,
        "source_blocks": source_blocks,
        "destination_blocks": destination_blocks,
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
        "positive_peer_failures": positive_failures,
        "expected_unused_local_cascade_failures": cascades,
        "cascade_explanation": (
            "The peer-only quick-start config has no KVCR local tier; unused local "
            "deposit/cascade failures are separate from peer delivery."
        ),
        "totals": dict(totals),
        "snapshots": parsed["snapshots"],
        "method": "Sum interval snapshots; do not subtract adjacent log lines.",
    }


def check_http_pair(
    seed: dict[str, Any], target: dict[str, Any], worker_id: str | int
) -> dict[str, Any]:
    """Validate independent HTTP evidence before interpreting transfer counters."""
    failures = []
    records = []
    for label, response, rank in (("seed", seed, 0), ("target", target, 1)):
        body = response.get("body") or {}
        identity = (body.get("nvext") or {}).get("worker_id") or {}
        cached = ((body.get("usage") or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        choices = body.get("choices") or []
        if response.get("http_status") != 200 or response.get("error"):
            failures.append(f"{label}: HTTP request did not succeed")
        if not choices:
            failures.append(f"{label}: missing choices")
        if identity.get("decode_worker_id") != worker_id:
            failures.append(f"{label}: wrong worker identity")
        actual_rank = identity.get("decode_dp_rank")
        if type(actual_rank) is not int or actual_rank != rank:
            failures.append(f"{label}: wrong DP rank {actual_rank!r}, expected {rank}")
        if type(cached) is not int or cached < 0:
            failures.append(f"{label}: missing or invalid cached_tokens")
        elif label == "seed" and cached != 0:
            failures.append("fresh seed unexpectedly reused cached tokens")
        elif label == "target" and (cached == 0 or cached % BLOCK_SIZE):
            failures.append("target cached_tokens must be positive and block aligned")
        records.append(
            {
                "stage": label,
                "cached_tokens": cached,
                "worker_id": identity.get("decode_worker_id"),
                "rank": actual_rank,
            }
        )
    return {"passed": not failures, "failures": failures, "stages": records}


def sanitize_logs(output: Path) -> None:
    """Redact signed download URL queries in artifact copies, never live logs."""
    records = []
    url_pattern = re.compile(r"https?://[^\s\"'<>]+")
    for path in sorted(output.glob("*.log")):
        original = path.read_bytes()
        redacted = 0

        def replace(match: re.Match) -> str:
            nonlocal redacted
            url = match.group(0)
            if "?" not in url:
                return url
            redacted += 1
            return url.split("?", 1)[0] + "?[REDACTED]"

        text = ANSI.sub("", original.decode(errors="replace"))
        sanitized = url_pattern.sub(replace, text).encode()
        path.write_bytes(sanitized)
        records.append(
            {
                "file": path.name,
                "original_sha256": hashlib.sha256(original).hexdigest(),
                "sanitized_sha256": hashlib.sha256(sanitized).hexdigest(),
                "redacted_url_queries": redacted,
            }
        )
    save_json(output / "log-sanitization.json", records)


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output_dir.resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise ValidationError(f"output directory must be empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + args.timeout
        self.children: list[tuple[str, subprocess.Popen, Any]] = []
        self.discovery: tempfile.TemporaryDirectory | None = None
        self.worker_log = args.worker_log or self.output / "worker.log"
        self.log_offset: int | None = None
        self.base_url = args.base_url or f"http://127.0.0.1:{args.http_port}"
        self.result: dict[str, Any] = {
            "started_at": utc_now(),
            "status": "running",
            "mode": "verify-only" if args.base_url else "launch-and-verify",
            "model": args.model,
            "base_url": self.base_url,
            "peer_transfer_verified": False,
            "requests": [],
            "limitations": [
                "One rank-0 to rank-1 mechanism test, not a performance benchmark.",
                "No semantic golden answer or independent KV-byte checksum is used.",
            ],
        }
        self.save()

    def save(self) -> None:
        save_json(self.output / "result.json", self.result)

    def remaining(self, limit: float) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ValidationError("overall test deadline expired")
        return min(limit, remaining)

    def pause(self, seconds: float) -> None:
        if self.remaining(seconds) < seconds:
            raise ValidationError("insufficient time remains before test deadline")
        time.sleep(seconds)

    def provenance(self) -> None:
        versions = {}
        for distribution in ("kvcr", "nixl", "vllm", "ai-dynamo", "ai-dynamo-runtime"):
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = None
        data = {
            "python": sys.version,
            "platform": platform.platform(),
            "versions": versions,
            "arguments": vars(self.args).copy(),
            "gpu_bindings": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES",
                    "SLURM_JOB_ID",
                )
            },
        }
        data["arguments"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in data["arguments"].items()
        }
        manifest = Path("/opt/kvcr-build/source-revisions.json")
        data["image_source_revisions"] = (
            json.loads(manifest.read_text()) if manifest.exists() else None
        )
        data["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        save_json(self.output / "provenance.json", data)

    def gpu_preflight(self) -> None:
        import torch

        memory = os.statvfs("/dev/shm")
        data = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "shm_available_bytes": memory.f_bavail * memory.f_frsize,
            "memlock_limits_bytes": resource.getrlimit(resource.RLIMIT_MEMLOCK),
            "devices": [],
        }
        save_json(self.output / "gpu-preflight.json", data)
        if not data["cuda_available"] or data["device_count"] != 2:
            raise ValidationError(
                "launch mode requires exactly two usable visible CUDA GPUs"
            )
        if data["shm_available_bytes"] < 8 * 1024**3:
            raise ValidationError(
                "launch mode requires at least 8 GiB free in /dev/shm"
            )
        for index in range(2):
            value = torch.ones(16, device=f"cuda:{index}").sum().item()
            data["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_probe": value,
                }
            )
            save_json(self.output / "gpu-preflight.json", data)
            if value != 16:
                raise ValidationError(f"CUDA compute probe failed on GPU {index}")

    def launch(self) -> None:
        self.gpu_preflight()
        for port in (self.args.http_port, 17771, 17772, 20081, 20082):
            check_port_available(port)
        self.discovery = tempfile.TemporaryDirectory(prefix="kvcr-e2e-")
        environment = os.environ.copy()
        environment.pop("NATS_SERVER", None)
        environment.update(
            DYN_REQUEST_PLANE="tcp",
            DYN_EVENT_PLANE="zmq",
            DYN_DISCOVERY_BACKEND="file",
            DYN_FILE_KV=self.discovery.name,
            PYTHONHASHSEED="0",
            VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES="1",
        )
        transfer = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": 2_000_000_000,
                "self_describing_kv_events": True,
                "secondary_tiers": [
                    {
                        "type": "kvcr",
                        "router_capabilities": ["router_hint"],
                        "control_host": "0.0.0.0",
                        "control_ports": [17771, 17772],
                        "control_advertise_host": "127.0.0.1",
                        "operation_timeout_ms": 5000,
                        "abandon_timeout_ms": 10000,
                        "enable_telemetry": True,
                    }
                ],
            },
        }
        events = {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": "tcp://*:20081",
            "enable_kv_cache_events": True,
        }
        save_json(self.output / "kv-transfer-config.json", transfer)
        save_json(self.output / "kv-events-config.json", events)
        commands = {
            "frontend": [
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--router-mode",
                "kv",
                "--http-port",
                str(self.args.http_port),
            ],
            "worker": [
                sys.executable,
                "-m",
                "dynamo.vllm",
                "--model",
                self.args.model,
                "--tensor-parallel-size",
                "1",
                "--data-parallel-size",
                "2",
                "--block-size",
                str(BLOCK_SIZE),
                "--max-model-len",
                "32000",
                "--enable-prefix-caching",
                "--disable-hybrid-kv-cache-manager",
                "--enforce-eager",
                "--gpu-memory-utilization",
                "0.8",
                "--kv-transfer-config",
                json.dumps(transfer),
                "--kv-events-config",
                json.dumps(events),
            ],
        }
        save_json(
            self.output / "launch.json",
            {
                "commands": commands,
                "discovery": self.discovery.name,
                "environment_overrides": {
                    key: environment[key]
                    for key in (
                        "DYN_REQUEST_PLANE",
                        "DYN_EVENT_PLANE",
                        "DYN_DISCOVERY_BACKEND",
                        "DYN_FILE_KV",
                        "PYTHONHASHSEED",
                        "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES",
                    )
                },
            },
        )
        for name, command in commands.items():
            child_environment = environment.copy()
            if name == "worker":
                child_environment["VLLM_LOGGING_LEVEL"] = "DEBUG"
            log = (self.output / f"{name}.log").open("wb")
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=child_environment,
                    start_new_session=True,
                )
            except BaseException:
                log.close()
                raise
            self.children.append((name, process, log))

    def check_children(self) -> None:
        for name, process, _ in self.children:
            if process.poll() is not None:
                raise ValidationError(
                    f"{name} exited early with status {process.returncode}"
                )

    def ready(self) -> None:
        deadline = time.monotonic() + self.remaining(self.args.readiness_timeout)
        last_error = "model not registered"
        while time.monotonic() < deadline:
            self.check_children()
            try:
                with urlopen(
                    self.base_url + "/v1/models", timeout=self.remaining(5)
                ) as response:
                    data = json.load(response)
                if any(
                    model.get("id") == self.args.model for model in data.get("data", [])
                ):
                    save_json(self.output / "models.json", data)
                    return
            except Exception as error:
                last_error = str(error)
            self.pause(1)
        raise ValidationError(f"model readiness timed out: {last_error}")

    def completion(
        self,
        stage: str,
        prompt: str,
        worker: str | int | None = None,
        rank: int | None = None,
    ) -> dict[str, Any]:
        self.check_children()
        headers = {"Content-Type": "application/json"}
        if worker is not None:
            headers["x-dynamo-worker-instance-id"] = str(worker)
        if rank is not None:
            headers["x-dynamo-dp-rank"] = str(rank)
        body = {
            "model": self.args.model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "seed": 0,
            "nvext": {"extra_fields": ["worker_id"]},
        }
        started_at = utc_now()
        save_json(
            self.output / f"{stage}.request.json",
            {
                "started_at": started_at,
                "url": self.base_url + "/v1/completions",
                "headers": headers,
                "body": body,
            },
        )
        started = time.monotonic()
        status, raw, response_body, error = None, b"", None, None
        response_headers = {}
        try:
            request = Request(
                self.base_url + "/v1/completions",
                headers=headers,
                data=json.dumps(body).encode(),
                method="POST",
            )
            try:
                response = urlopen(
                    request, timeout=self.remaining(self.args.request_timeout)
                )
            except HTTPError as failure:
                response = failure
            with response:
                status = response.status
                response_headers = dict(response.headers.items())
                raw = response.read()
            response_body = json.loads(raw)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        result = {
            "stage": stage,
            "started_at": started_at,
            "finished_at": utc_now(),
            "latency_seconds": time.monotonic() - started,
            "http_status": status,
            "headers": response_headers,
            "body": response_body,
            "error": error,
        }
        if response_body is None:
            result["body_text"] = raw.decode(errors="replace")
        save_json(self.output / f"{stage}.response.json", result)
        self.result["requests"].append(
            {
                "stage": stage,
                "started_at": started_at,
                "finished_at": result["finished_at"],
                "request_file": f"{stage}.request.json",
                "response_file": f"{stage}.response.json",
                "http_status": status,
            }
        )
        self.save()
        return result

    def verify(self) -> None:
        if not self.worker_log.is_file():
            raise ValidationError(f"worker log does not exist: {self.worker_log}")
        original = self.worker_log.read_bytes()
        # Retain any partial line and its immediately preceding engine context.
        offset = log_window_boundary(original)
        self.log_offset = offset
        line_offset = original[:offset].count(b"\n")
        stat = self.worker_log.stat()
        identity = (stat.st_dev, stat.st_ino)
        self.result["worker_log_window"] = {
            "path": str(self.worker_log),
            "start_byte": offset,
            "start_line": line_offset + 1,
            "started_at": utc_now(),
        }
        self.save()
        probe = self.completion("01-probe", "worker identity probe")
        if probe["http_status"] != 200 or not isinstance(probe["body"], dict):
            raise ValidationError("worker identity probe failed; see response evidence")
        worker = ((probe["body"].get("nvext") or {}).get("worker_id") or {}).get(
            "decode_worker_id"
        )
        if type(worker) not in (str, int) or str(worker) == "":
            raise ValidationError("worker identity probe returned no usable worker ID")
        tag = uuid.uuid4().hex
        prompt = " ".join(f"{tag} section{i} cache routing data" for i in range(80))
        self.result["prefix_uuid"] = tag
        seed = self.completion("02-seed-rank0", prompt, worker, 0)
        self.pause(self.args.index_wait)
        target = self.completion("03-target-rank1", prompt, worker, 1)
        http = check_http_pair(seed, target, worker)
        self.result["http_validation"] = http
        self.save()
        if not http["passed"]:
            raise ValidationError("; ".join(http["failures"]))
        seed_text = seed["body"]["choices"][0].get("text")
        target_text = target["body"]["choices"][0].get("text")
        self.result["output_consistency"] = {
            "checked": self.args.check_output_consistency,
            "equal": seed_text == target_text,
            "seed_sha256": hashlib.sha256(seed_text.encode()).hexdigest()
            if isinstance(seed_text, str)
            else None,
            "target_sha256": hashlib.sha256(target_text.encode()).hexdigest()
            if isinstance(target_text, str)
            else None,
            "semantic_correctness_checked": False,
        }
        expected = http["stages"][1]["cached_tokens"] // BLOCK_SIZE
        deadline = time.monotonic() + self.remaining(self.args.telemetry_timeout)
        telemetry = None
        while time.monotonic() < deadline:
            self.check_children()
            stat = self.worker_log.stat()
            if (stat.st_dev, stat.st_ino) != identity or stat.st_size < offset:
                raise ValidationError(
                    "worker log rotated or truncated during verification"
                )
            with self.worker_log.open("rb") as stream:
                stream.seek(offset)
                window = stream.read()
            (self.output / "worker-window.log").write_bytes(window)
            parsed = parse_metrics(
                window.decode(errors="replace"), line_offset=line_offset
            )
            telemetry = assess_metrics(
                parsed, expected, retrieval_started_at=target["started_at"]
            )
            save_json(self.output / "telemetry.json", telemetry)
            if telemetry["status"] == "failed":
                raise ValidationError("; ".join(telemetry["failures"]))
            if telemetry["status"] == "passed":
                self.result["peer_transfer_verified"] = True
                self.result["transfer"] = {
                    key: telemetry[key]
                    for key in (
                        "source_blocks",
                        "destination_blocks",
                        "source_bytes",
                        "destination_bytes",
                    )
                }
                break
            self.pause(1)
        else:
            if telemetry is not None:
                telemetry = assess_metrics(
                    parsed,
                    expected,
                    retrieval_started_at=target["started_at"],
                    final=True,
                )
                save_json(self.output / "telemetry.json", telemetry)
            missing = telemetry["missing"] if telemetry else ["no metric snapshots"]
            failures = telemetry["failures"] if telemetry else []
            raise ValidationError(
                f"telemetry deadline expired; missing proof: {missing}; "
                f"failures: {failures}"
            )
        if self.args.check_output_consistency and (
            not isinstance(seed_text, str) or seed_text != target_text
        ):
            raise ValidationError(
                "cold and cached outputs differ; peer telemetry passed separately"
            )

    def cleanup(self) -> None:
        outcomes = []
        for _, process, _ in self.children:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 20
        for name, process, log in self.children:
            killed = False
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                killed = True
            # Also terminate descendants if the original leader already exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
                killed = True
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                outcomes.append(
                    {"name": name, "pid": process.pid, "error": "cleanup timed out"}
                )
            else:
                outcomes.append(
                    {
                        "name": name,
                        "pid": process.pid,
                        "exit_code": process.returncode,
                        "kill_sent": killed,
                    }
                )
            log.close()
        if self.discovery is not None:
            self.discovery.cleanup()
        save_json(self.output / "cleanup.json", {"owned_processes": outcomes})
        if any("error" in outcome for outcome in outcomes):
            raise ValidationError("owned process cleanup did not finish")

    def capture_log_tail(self) -> None:
        if self.log_offset is None or not self.worker_log.is_file():
            return
        with self.worker_log.open("rb") as stream:
            stream.seek(self.log_offset)
            window = stream.read()
        (self.output / "worker-window.log").write_bytes(window)
        self.result["worker_log_window"].update(
            end_byte=self.log_offset + len(window), finished_at=utc_now()
        )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base-url", help="Existing endpoint; requires --worker-log")
    parser.add_argument(
        "--worker-log", type=Path, help="Existing worker log; requires --base-url"
    )
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument(
        "--timeout", type=float, default=900, help="Overall deadline in seconds"
    )
    parser.add_argument("--readiness-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--telemetry-timeout", type=float, default=60)
    parser.add_argument("--index-wait", type=float, default=15)
    parser.add_argument("--check-output-consistency", action="store_true")
    args = parser.parse_args()
    if bool(args.base_url) != bool(args.worker_log):
        parser.error("--base-url and --worker-log must be supplied together")
    if args.base_url:
        parts = urlsplit(args.base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            parser.error("--base-url must be an HTTP(S) URL")
        if parts.username or parts.password or parts.query or parts.fragment:
            parser.error("--base-url must not contain credentials, query, or fragment")
        args.base_url = args.base_url.rstrip("/")
    for name in (
        "timeout",
        "readiness_timeout",
        "request_timeout",
        "telemetry_timeout",
    ):
        if not 0 < getattr(args, name) <= 7200:
            parser.error(f"--{name.replace('_', '-')} must be >0 and <=7200 seconds")
    if not 0 <= args.index_wait <= 300 or not 1 <= args.http_port <= 65535:
        parser.error("invalid index wait or HTTP port")
    return args


def main() -> int:
    args = arguments()
    try:
        runner = Runner(args)
    except Exception as error:
        print(f"Cannot initialize test: {error}", file=sys.stderr)
        return 1

    def interrupted(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        runner.provenance()
        if not args.base_url:
            runner.launch()
        runner.ready()
        runner.verify()
        runner.result["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        runner.result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            runner.cleanup()
        except Exception as error:
            runner.result.update(status="failed", cleanup_error=str(error))
        try:
            runner.capture_log_tail()
            sanitize_logs(runner.output)
        except Exception as error:
            runner.result.update(status="failed", artifact_error=str(error))
        runner.result["finished_at"] = utc_now()
        runner.save()
        try:
            write_junit(runner.output, runner.result)
        except Exception as error:
            runner.result.update(status="failed", report_error=str(error))
            runner.save()
        for number, handler in previous.items():
            signal.signal(number, handler)
    print(
        json.dumps(
            {
                "status": runner.result["status"],
                "peer_transfer_verified": runner.result["peer_transfer_verified"],
                "transfer": runner.result.get("transfer"),
                "error": runner.result.get("error"),
                "result": str(runner.output / "result.json"),
            }
        )
    )
    return 0 if runner.result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
