# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of pinned model inputs to the separate DP=2 smoke."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "e2e" / "run_vllm_kvcr.py"
SPEC = importlib.util.spec_from_file_location("vllm_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_snapshot_input_preserves_canonical_served_model(tmp_path):
    args = smoke.arguments(
        ["--model", "Qwen/Qwen3-0.6B", "--model-path", str(tmp_path)]
    )
    command = smoke.worker_arguments(args)
    assert command[command.index("--model") + 1] == str(tmp_path.resolve())
    assert command[command.index("--served-model-name") + 1] == "Qwen/Qwen3-0.6B"
    assert command[command.index("--data-parallel-size") + 1] == "2"


def test_smoke_rejects_missing_local_snapshot(tmp_path):
    with pytest.raises(SystemExit):
        smoke.arguments(["--model-path", str(tmp_path / "missing")])


def test_existing_manual_model_input_remains_available():
    args = smoke.arguments(["--model", "custom/model"])
    command = smoke.worker_arguments(args)
    assert command[command.index("--model") + 1] == "custom/model"
    assert args.model_path is None


def test_owned_worker_is_reaped_and_cleanup_evidence_written(tmp_path):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        outcomes = []
        smoke.stop_owned("worker", process, outcomes)
        smoke.write_cleanup(tmp_path, outcomes)
        assert process.poll() is not None
        report = json.loads((tmp_path / "cleanup.json").read_text())
        assert report["status"] == "PASS"
        assert report["owned_processes"][0]["pid"] == process.pid
        assert report["duration_seconds"] > 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_cleanup_failure_cannot_be_reported_as_pass(tmp_path):
    smoke.write_cleanup(tmp_path, [{"name": "worker", "error": "did not exit"}])
    assert json.loads((tmp_path / "cleanup.json").read_text())["status"] == "FAIL"


def test_launch_receipt_records_actual_owned_process_and_safe_environment(tmp_path):
    command = [sys.executable, "-c", "print('receipt-ready', flush=True)"]
    with (tmp_path / "test.log").open("wb") as log:
        process = smoke.start_recorded_process(
            "worker",
            command,
            {"CUDA_VISIBLE_DEVICES": "0,1", "SECRET": "do-not-record"},
            log,
            tmp_path,
        )
        process.wait(timeout=5)
    receipt = json.loads((tmp_path / "worker.launch.json").read_text())
    assert receipt["pid"] == process.pid
    assert receipt["command"] == command
    assert receipt["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "SECRET" not in receipt["environment"]
    assert "receipt-ready" in (tmp_path / "test.log").read_text().splitlines()
