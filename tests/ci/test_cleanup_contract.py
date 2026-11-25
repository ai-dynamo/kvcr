# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU regression checks for owned cleanup and interruption evidence."""

import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

import pytest

E2E = Path(__file__).parents[1] / "e2e"
sys.path.insert(0, str(E2E))

from phase1 import stack  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "cleanup_phase1_runner", E2E / "run_phase1.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("interrupt_before_cleanup", [False, True])
def test_repeated_signals_preserve_cleanup_and_failure_result(
    tmp_path, monkeypatch, interrupt_before_cleanup
):
    args = runner.arguments(["--output-dir", str(tmp_path / "run"), "--cases", "TC-01"])
    run = runner.Runner(args)
    monkeypatch.setattr(runner, "arguments", lambda: args)
    monkeypatch.setattr(runner, "Runner", lambda _args: run)
    monkeypatch.setattr(runner, "provenance", lambda _args: {})

    def preflight():
        if interrupt_before_cleanup:
            signal.raise_signal(signal.SIGTERM)
        return {}

    monkeypatch.setattr(run.stack, "preflight", preflight)
    monkeypatch.setattr(run, "reference_phase", lambda: None)
    monkeypatch.setattr(run, "stack_phase", lambda: None)

    def passing_case(_client):
        run.result["cases"].append(
            {"case": "TC-01", "title": "test", "status": "PASS", "checks": []}
        )

    monkeypatch.setattr(run, "cases_phase", passing_case)
    cleanup = run.stack.cleanup

    def interrupted_cleanup():
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGTERM)
        return cleanup()

    monkeypatch.setattr(run.stack, "cleanup", interrupted_cleanup)
    previous = {
        number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        returncode = runner.main()
    except KeyboardInterrupt:
        pytest.fail("a teardown signal interrupted owned cleanup")
    assert returncode == 1
    result = json.loads((run.output / "result.json").read_text())
    assert result["status"] == "FAIL"
    assert "KeyboardInterrupt" in result["error"]
    assert (run.output / "stack/cleanup.json").is_file()
    assert all(
        signal.getsignal(number) == handler for number, handler in previous.items()
    )


@pytest.mark.parametrize("ignore_term", [False, True])
def test_real_process_stop_distinguishes_grace_expiry(tmp_path, ignore_term):
    ready = tmp_path / "ready"
    handler = "signal.SIG_IGN" if ignore_term else "lambda *_: sys.exit(0)"
    program = (
        "import pathlib,signal,sys,time; "
        f"signal.signal(signal.SIGTERM,{handler}); "
        f"pathlib.Path({str(ready)!r}).touch(); time.sleep(30)"
    )
    child = stack.Managed(
        "worker-0", [sys.executable, "-c", program], {}, tmp_path / "worker.log"
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and child.alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), child.log_text()
        outcome = child.stop(grace=0.05)
        assert outcome["grace_expired"] is ignore_term
        assert outcome["classification"] == (
            "forced_kill" if ignore_term else "graceful_exit"
        )
        assert outcome["intent"] == "normal_shutdown"
        assert outcome["leader_exited"] is True
        assert outcome["duration_seconds"] >= 0
        assert outcome["signals"][0]["signal"] == "SIGTERM"
        assert outcome["signals"][-1]["signal"] == "SIGKILL"
        assert child.stopped
    finally:
        if child.alive():
            child.process.kill()
            child.process.wait(timeout=5)


def test_owned_path_removal_failure_is_reported_and_blocks_cleanup(
    tmp_path, monkeypatch
):
    owned = stack.Stack(stack.Settings(model="m", output=tmp_path), tmp_path / "stack")
    resource = owned.kvcr_run_dir / "leftover.sock"
    resource.touch()
    unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path == resource:
            raise PermissionError("owned socket could not be removed")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        with pytest.raises(stack.ValidationError, match="cleanup"):
            owned.cleanup()
    report = json.loads((owned.evidence / "cleanup.json").read_text())
    assert report["status"] == "FAIL"
    assert report["reclaimed"] is False
    assert any(
        item["status"] == "FAIL" and item["path"] == str(owned.kvcr_run_dir)
        for item in report["owned_paths"]
    )
    resource.unlink()
    if owned.kvcr_run_dir.exists():
        owned.kvcr_run_dir.rmdir()


def test_injected_failure_is_not_reported_as_graceful_shutdown(tmp_path):
    owned = stack.Stack(stack.Settings(model="m", output=tmp_path), tmp_path / "stack")
    child = stack.Managed(
        "worker-0",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        {},
        tmp_path / "worker.log",
    )
    owned.children.append(child)
    owned.workers[0] = child
    try:
        owned.kill_worker(0)
        assert child._log.closed
        report = owned.cleanup()
        injection = report["owned_processes"][0]
        assert injection["action"] == "injected_sigkill"
        assert injection["classification"] == "expected_injected_kill"
        assert report["graceful"] is None
        assert report["observation_complete"] is False
        assert report["warnings"] == []
    finally:
        if child.alive():
            child.process.kill()
            child.process.wait(timeout=5)


def test_directory_enumeration_failure_still_writes_cleanup_evidence(
    tmp_path, monkeypatch
):
    owned = stack.Stack(stack.Settings(model="m", output=tmp_path), tmp_path / "stack")
    original = Path.rglob

    def denied(path, pattern):
        if path == owned.kvcr_run_dir:
            raise PermissionError("runtime directory cannot be enumerated")
        return original(path, pattern)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rglob", denied)
        with pytest.raises(stack.ValidationError, match="cleanup"):
            owned.cleanup()
    report = json.loads((owned.evidence / "cleanup.json").read_text())
    assert report["status"] == "FAIL"
    assert any(
        "cannot be enumerated" in error
        for item in report["owned_paths"]
        for error in item.get("errors", [])
    )
    if owned.kvcr_run_dir.exists():
        owned.kvcr_run_dir.rmdir()
