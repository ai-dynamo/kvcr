# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch one catalog profile with immutable inputs and a fail-closed envelope.

Each invocation owns a fresh output directory. The harness owns its serving
processes; this wrapper supervises prefetch and the harness, preserving evidence
on setup errors, timeouts and handled termination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ci_cleanup import validate_phase1_cleanup
from ci_identity import verify_build_identity
from ci_integrations import get_integration
from ci_model_access import prepare_model, validate_snapshot
from ci_profiles import SMOKE_ID, fingerprint, get_profile, load_catalog
from ci_resources import observe_gpu_processes, observe_ports
from phase1.common import validate_gpu_budget

E2E = Path(__file__).resolve().parent
REPO = E2E.parents[1]
BUILD_METADATA = Path("/opt/kvcr-build-metadata.json")


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def harness_fingerprint(smoke: bool = False) -> str:
    paths = (
        [E2E / "run_vllm_kvcr.py"]
        if smoke
        else sorted((E2E / "phase1").glob("*.py")) + [E2E / "run_phase1.py"]
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_identity(expected: str, diagnostic: bool = False) -> dict:
    checkout_error = None
    try:
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        if not diagnostic:
            raise ValueError("cannot verify checked-out recipe source") from error
        actual = None
        checkout_error = f"unavailable: {type(error).__name__}"
    metadata = json.loads(BUILD_METADATA.read_text()) if BUILD_METADATA.exists() else {}
    installed = metadata.get("kvcr_sha")
    recipe_sha = os.environ.get("KVCR_CI_RECIPE_SHA")
    if recipe_sha and not diagnostic:
        verify_build_identity(
            metadata,
            source_sha=expected,
            recipe_sha=recipe_sha,
            image_profile=os.environ["KVCR_CI_IMAGE_PROFILE"],
            resolved_inputs_sha256=os.environ["KVCR_CI_INPUTS_SHA256"],
        )
    mismatches = []
    recipe = os.environ.get("KVCR_CI_RECIPE_SHA") or os.environ.get(
        "GITHUB_SHA", expected
    )
    if actual != recipe:
        mismatches.append(f"checked-out recipe {actual} differs from expected {recipe}")
    if installed != expected:
        mismatches.append(
            f"installed source {installed!r} differs from expected {expected}"
        )
    if mismatches and not diagnostic:
        raise ValueError("; ".join(mismatches))
    return {
        "source_sha": expected,
        "checked_out_source_sha": actual,
        "checked_out_source_error": checkout_error,
        "workflow_sha": os.environ.get("GITHUB_SHA", actual),
        "installed_source_sha": installed,
        "image_build_metadata": metadata,
        "source_mismatches": mismatches,
    }


def owned_processes(token: str) -> dict[int, str]:
    """Find this invocation's inherited marker without relying on sessions."""
    marker = f"KVCR_CI_OWNER_TOKEN={token}".encode()
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            if marker not in (directory / "environ").read_bytes().split(b"\0"):
                continue
            fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            if fields[0] != "Z":
                found[int(directory.name)] = fields[19]  # Linux starttime, field 22.
        except (OSError, ValueError, IndexError):
            continue
    return found


def signal_owned(processes: dict[int, str], number: int) -> None:
    for pid, started in processes.items():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            # Do not signal a PID that has been reused since enumeration.
            if fields[19] == started and fields[0] != "Z":
                os.kill(pid, number)
        except (ProcessLookupError, FileNotFoundError):
            continue


def reap_owned_sessions(token: str) -> dict:
    """Bound detached-worker cleanup after cooperative harness cleanup fails."""
    initial = owned_processes(token)
    remaining = initial
    for number, grace in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 2.0)):
        if not remaining:
            break
        signal_owned(remaining, number)
        deadline = time.monotonic() + grace
        while remaining and time.monotonic() < deadline:
            time.sleep(0.05)
            remaining = owned_processes(token)
    return {
        "status": "FAIL" if remaining else "PASS",
        "required_fallback": bool(initial),
        "terminated_pids": sorted(initial.keys() - remaining.keys()),
        "surviving_pids": sorted(remaining),
    }


def run_process(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout: float,
    grace: float = 180,
) -> dict:
    """Bound one owned process group, requesting cooperative harness cleanup first."""
    started = time.monotonic()
    interrupted = False
    timed_out = False
    forced = False
    owner = uuid.uuid4().hex
    child_env = {**env, "KVCR_CI_OWNER_TOKEN": owner}
    ownership = {"status": "NOT_RUN", "required_fallback": False}
    process: subprocess.Popen | None = None

    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous = {
        number: signal.signal(number, request_stop)
        for number in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                if interrupted or time.monotonic() - started >= timeout:
                    timed_out = not interrupted
                    # Both harnesses handle SIGTERM by unwinding their own
                    # separately-sessioned serving workers before exiting.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        forced = True
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    break
                time.sleep(0.05)
            returncode = process.wait()
    finally:
        try:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            # The serving harness deliberately creates separate sessions for
            # workers. A process-group kill alone cannot contain those workers.
            ownership = reap_owned_sessions(owner)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return {
        "status": "PASS"
        if returncode == 0
        and not interrupted
        and not timed_out
        and ownership["status"] == "PASS"
        and not ownership["required_fallback"]
        else "FAIL",
        "returncode": returncode,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "forced_kill": forced,
        "owned_process_cleanup": ownership,
        "duration_seconds": time.monotonic() - started,
    }


def validate_snapshot_override(profile, path, diagnostic):
    if not diagnostic:
        raise ValueError("a supplied model snapshot requires --diagnostic")
    validate_snapshot(profile, path)
    return path.resolve()


def prepare_snapshot(profile, output, env, override=None, diagnostic=False):
    return prepare_model(profile, output, env, run_process, override, diagnostic)


def build_command(profile, output, snapshot, prompt_tag, gpus=None):
    return get_integration(profile["integration"]).build_command(
        profile, output, snapshot, prompt_tag, gpus
    )


def policy_fingerprint() -> str:
    paths = [
        E2E / name
        for name in (
            "ci_profiles.py",
            "ci_models.json",
            "ci_profiles.json",
            "ci_model_access.py",
            "ci_identity.py",
            "ci_resources.py",
            "ci_cleanup.py",
            "run_ci_profile.py",
            "summarize_ci.py",
        )
    ] + sorted((E2E / "ci_integrations").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(E2E)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def gpu_observation() -> dict:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        response = subprocess.check_output(
            command, text=True, stderr=subprocess.PIPE, timeout=10
        )
        rows = []
        for line in response.splitlines():
            index, gpu_uuid, name, memory = (
                part.strip() for part in line.split(",", 3)
            )
            rows.append(
                {
                    "index": int(index),
                    "uuid": gpu_uuid,
                    "name": name,
                    "memory_total_mib": int(memory),
                }
            )
        return {"status": "OBSERVED", "allocated_devices": rows}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"status": "UNOBSERVED", "error": type(error).__name__}


def cleanup_evidence(output: Path, smoke: bool) -> dict:
    path = output / "run" / ("cleanup.json" if smoke else "stack/cleanup.json")
    report = json.loads(path.read_text())
    if not smoke:
        validate_phase1_cleanup(report)
    if not isinstance(report, dict) or not isinstance(
        report.get("owned_processes"), list
    ):
        raise ValueError("missing owned-process cleanup evidence")
    if report.get("status", "PASS") != "PASS" or any(
        not isinstance(entry, dict) or "error" in entry
        for entry in report["owned_processes"]
    ):
        raise RuntimeError("harness cleanup reports surviving processes or errors")
    return report


def run_profile(
    profile: dict,
    output: Path,
    image: str,
    expected_source_sha: str,
    run_id: str,
    run_attempt: str,
    *,
    diagnostic: bool = False,
    gpus: list[int] | None = None,
    snapshot_override: Path | None = None,
) -> int:
    if output.exists() and any(output.iterdir()):
        raise ValueError("profile output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 2,
        "recipe_sha": os.environ.get(
            "KVCR_CI_RECIPE_SHA", os.environ.get("GITHUB_SHA", expected_source_sha)
        ),
        "image_profile": os.environ.get(
            "KVCR_CI_IMAGE_PROFILE", profile["integration"]
        ),
        "resolved_inputs_sha256": os.environ.get("KVCR_CI_INPUTS_SHA256", ""),
        "policy_sha256": policy_fingerprint(),
        "profile_id": profile["id"],
        "profile_fingerprint": fingerprint(profile),
        "run_id": str(run_id),
        "run_attempt": str(run_attempt),
        "source_sha": expected_source_sha,
        "image": image,
        "diagnostic": diagnostic,
        "status": "FAIL",
        "started_at": timestamp(),
        # Retained schema alias for historical readers; this is a requested mapping.
        "actual_gpus": gpus if gpus is not None else profile["resources"]["gpus"],
        "requested_gpu_mapping": gpus
        if gpus is not None
        else profile["resources"]["gpus"],
        "gpu_observation": gpu_observation(),
        "intended_gpus": profile["resources"]["gpus"],
        "harness_sha256": harness_fingerprint(profile["id"] == SMOKE_ID),
        **{
            key: profile[key]
            for key in (
                "integration",
                "model_family",
                "model_id",
                "model_revision",
                "topology",
                "mode",
                "cases",
            )
        },
        "setup": {"status": "NOT_RUN", "duration_seconds": 0},
        "harness": {"status": "NOT_RUN", "duration_seconds": 0},
        "cleanup": {"status": "NOT_RUN", "duration_seconds": 0},
    }
    save_json(output / "profile-result.json", result)
    started = time.monotonic()
    stage = "setup"
    resource_policy = get_integration(profile["integration"]).runtime_resources(profile)
    result["resource_observations"] = {
        "declared": resource_policy,
        "before_ports": observe_ports(resource_policy["tcp_ports"]),
        "before_gpu_processes": observe_gpu_processes(),
    }
    env = os.environ.copy()
    env.update(
        UCX_TLS="tcp,sm,self,cuda_copy",
        KVCR_TEST_IMAGE=image,
        DYN_ROUTER_WORKER_SELECTION_POLICY="kvcr",
    )
    env.update(get_integration(profile["integration"]).environment(profile))
    try:
        if result["resource_observations"]["before_ports"]["status"] != "AVAILABLE":
            raise RuntimeError("declared runtime ports are unavailable before setup")
        result.update(
            source_identity(expected_source_sha, diagnostic=True)
            if diagnostic
            else source_identity(expected_source_sha)
        )
        # Retain image provenance beside results, without relying on later image access.
        metadata_dir = output / "image-provenance"
        metadata_dir.mkdir(exist_ok=True)
        for installed_record in Path("/opt").glob("kvcr-*-*.json"):
            if installed_record.name in {
                "kvcr-build-metadata.json",
                "kvcr-dynamo-install-report.json",
                "kvcr-install-report.json",
            }:
                (metadata_dir / installed_record.name).write_bytes(
                    installed_record.read_bytes()
                )
        snapshot = prepare_snapshot(profile, output, env, snapshot_override, diagnostic)
        result["model_path"] = str(snapshot)
        result["setup"] = {
            "status": "PASS",
            "duration_seconds": time.monotonic() - started,
        }
        stage = "harness"
        prompt_tag = f"ci_{profile['id']}_{run_id}_{run_attempt}".replace("-", "_")
        command = build_command(profile, output, snapshot, prompt_tag, gpus)
        result["command"] = command
        result["environment"] = {
            key: env[key]
            for key in (
                "UCX_TLS",
                "DYN_ROUTER_WORKER_SELECTION_POLICY",
                "VLLM_BATCH_INVARIANT",
            )
            if key in env
        }
        save_json(output / "profile-result.json", result)
        result["harness"] = run_process(
            command, env, output / "console.log", profile["launch"]["timeout"] + 120
        )
        native = json.loads((output / "run/result.json").read_text())
        expected_verdict = "success" if profile["id"] == SMOKE_ID else "PASS"
        result["status"] = (
            "PASS"
            if native.get("status") == expected_verdict
            and result["harness"]["status"] == "PASS"
            else "BLOCKED"
            if native.get("status") == "BLOCKED"
            else "FAIL"
        )
        result["phase_durations_seconds"] = native.get("phase_durations_seconds", {})
    except (Exception, KeyboardInterrupt) as error:
        result["status"] = "FAIL"
        result["error"] = f"{type(error).__name__}: {error}"
        result[stage]["status"] = "FAIL"
        if stage == "setup":
            result["setup"]["duration_seconds"] = time.monotonic() - started
    finally:
        cleanup_started = time.monotonic()
        actual_cleanup_seconds = 0
        try:
            if stage == "harness":
                result["cleanup"]["evidence"] = cleanup_evidence(
                    output, profile["id"] == SMOKE_ID
                )
                actual_cleanup_seconds = result.get("phase_durations_seconds", {}).get(
                    "cleanup", result["cleanup"]["evidence"].get("duration_seconds", 0)
                )
            result["cleanup"]["status"] = "PASS"
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            result["cleanup"].update(status="FAIL", error=str(error))
            result["status"] = "FAIL"
        verification_seconds = time.monotonic() - cleanup_started
        result["cleanup"]["verification_duration_seconds"] = verification_seconds
        result["cleanup"]["duration_seconds"] = (
            actual_cleanup_seconds + verification_seconds
        )
        observations = result["resource_observations"]
        observations["after_ports"] = observe_ports(resource_policy["tcp_ports"])
        observations["after_gpu_processes"] = observe_gpu_processes()
        if observations["after_ports"]["status"] != "AVAILABLE":
            result["cleanup"].update(
                status="FAIL", error="declared runtime port remains unavailable"
            )
            result["status"] = "FAIL"
        result["finished_at"] = timestamp()
        result["duration_seconds"] = time.monotonic() - started
        save_json(output / "profile-result.json", result)
    return 0 if result["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", default="vllm")
    parser.add_argument("--recipe-sha", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--image-profile", default="")
    parser.add_argument("--resolved-inputs-sha256", default="")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-source-sha", "--source-sha", required=True)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument(
        "--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="explicitly nonqualifying local diagnostic",
    )
    parser.add_argument("--gpus", help="diagnostic-only GPU mapping override")
    parser.add_argument(
        "--model-path", type=Path, help="diagnostic-only existing pinned snapshot"
    )
    args = parser.parse_args()
    try:
        catalog = load_catalog(integration=args.integration)
        profile = get_profile(catalog, args.profile)
        adapter = get_integration(args.integration)
        required = adapter.required_profile_ids(catalog) | {
            item["id"] for item in adapter.smoke_profiles(catalog)
        }
        if profile["id"] not in required and not args.diagnostic:
            raise ValueError("retired/manual profiles require --diagnostic")
        if not args.diagnostic and not args.dry_run:
            if not re.fullmatch(r"[a-f0-9]{40}", args.recipe_sha) or not re.fullmatch(
                r"[a-f0-9]{64}", args.resolved_inputs_sha256
            ):
                raise ValueError(
                    "production execution requires recipe SHA and "
                    "resolved build input digest"
                )
            if not args.image_profile:
                raise ValueError("production execution requires image profile")
        os.environ.update(
            KVCR_CI_RECIPE_SHA=args.recipe_sha,
            KVCR_CI_IMAGE_PROFILE=args.image_profile or args.integration,
            KVCR_CI_INPUTS_SHA256=args.resolved_inputs_sha256,
        )
        if not args.diagnostic and not re.fullmatch(
            r".+@sha256:[0-9a-f]{64}", args.image
        ):
            raise ValueError("CI profiles require an immutable image digest")
        if (
            not re.fullmatch(r"[0-9a-f]{40}", args.expected_source_sha)
            or not args.run_id
            or not args.run_attempt
        ):
            raise ValueError("source SHA and run identity are required")
        gpus = None
        if args.gpus or args.model_path:
            if not args.diagnostic:
                raise ValueError("GPU/model snapshot overrides require --diagnostic")
        if args.gpus:
            gpus = [int(value) for value in args.gpus.split(",")]
            if (
                profile["id"] == SMOKE_ID
                or len(gpus) != len(profile["resources"]["gpus"])
                or any(gpu < 0 for gpu in gpus)
            ):
                raise ValueError("invalid diagnostic GPU mapping")
            validate_gpu_budget(gpus, profile["resources"]["gpu_memory_utilization"])
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "profile_id": profile["id"],
                        "profile_fingerprint": fingerprint(profile),
                        "diagnostic": args.diagnostic,
                        "model_revision": profile["model_revision"],
                        "command": build_command(
                            profile,
                            args.output_dir,
                            args.model_path or Path("<pinned-snapshot>"),
                            "<run-tag>",
                            gpus,
                        ),
                    }
                )
            )
            return 0
        return run_profile(
            profile,
            args.output_dir.resolve(),
            args.image,
            args.expected_source_sha,
            args.run_id,
            args.run_attempt,
            diagnostic=args.diagnostic,
            gpus=gpus,
            snapshot_override=args.model_path,
        )
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
