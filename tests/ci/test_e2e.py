# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU evidence-validation tests; these never claim a real transfer occurred."""

import importlib.util
import json
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "e2e" / "run_vllm.py"
_SPEC = importlib.util.spec_from_file_location("kvcr_e2e_runner", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

START = "2026-09-16T22:42:09+00:00"


def sample(engine, metrics, timestamp="2026-09-16T22:42:13Z"):
    values = ", ".join(f"{key}={value}" for key, value in metrics.items())
    return (
        f"{timestamp} INFO loggers.log: Engine {engine:03d}: statistics\n"
        f"{timestamp} INFO metrics.log: KV Transfer metrics: {values}\n"
    )


def successful_log(blocks=5, byte_count=1000):
    return sample(
        0,
        {
            runner.TRANSFER_SUCCESS: 1,
            runner.SOURCE_SUCCESS: 1,
            runner.SOURCE_BLOCKS: blocks,
            runner.SOURCE_BYTES: byte_count,
        },
    ) + sample(
        1,
        {
            runner.DESTINATION_SUCCESS: 1,
            runner.DESTINATION_BLOCKS: blocks,
            runner.DESTINATION_BYTES: byte_count,
        },
    )


def assess(text, expected=5, final=False):
    return runner.assess_metrics(
        runner.parse_metrics(text),
        expected,
        retrieval_started_at=START,
        final=final,
    )


def test_reset_intervals_are_summed_not_differenced():
    first = successful_log(blocks=2, byte_count=200)
    second = successful_log(blocks=3, byte_count=300)
    result = assess(first + second)
    assert result["status"] == "passed"
    assert result["source_blocks"] == result["destination_blocks"] == 5
    assert result["source_bytes"] == result["destination_bytes"] == 500


def test_partial_reporting_waits_for_remaining_intervals():
    first = successful_log(blocks=2, byte_count=200)
    assert assess(first)["status"] == "incomplete"
    assert assess(first, final=True)["status"] == "failed"


def test_zero_interim_samples_wait_for_positive_completion():
    empty = successful_log(blocks=0, byte_count=0)
    empty = empty.replace("_count=1", "_count=0")
    assert assess(empty)["status"] == "incomplete"
    assert assess(empty + successful_log())["status"] == "passed"


@pytest.mark.parametrize("outcome", ["failed", "partial"])
def test_positive_delivery_failure_disqualifies_success(outcome):
    metric = f"vllm:kvcr_duration_seconds:('remote_deliver', '{outcome}')_count"
    result = assess(successful_log() + sample(1, {metric: 1}))
    assert result["status"] == "failed"
    assert result["positive_peer_failures"][0]["metric"] == metric


def test_expected_local_cascade_failure_is_reported_separately():
    result = assess(successful_log() + sample(0, {runner.CASCADE_FAILURE: 1}))
    assert result["status"] == "passed"
    assert result["positive_peer_failures"] == []
    assert result["expected_unused_local_cascade_failures"][0]["value"] == 1


def test_missing_metrics_remain_absent_not_fabricated_zero():
    result = assess("")
    assert result["status"] == "incomplete"
    assert result["source_blocks"] is None
    assert result["destination_bytes"] is None
    assert result["missing"]


def test_missing_engine_context_prevents_attribution():
    lines = successful_log().splitlines(keepends=True)
    result = assess("".join(lines[1:]))
    assert result["status"] == "failed"
    assert "engine context" in result["failures"][0]


def test_boundary_keeps_engine_header_before_partial_metrics():
    text = successful_log().encode()
    first_newline = text.index(b"\n") + 1
    prefix = b"older unrelated log line\n"
    original = prefix + text[: first_newline + 12]
    offset = runner.log_window_boundary(original)
    assert offset == len(prefix)
    assert runner.parse_metrics((prefix + text)[offset:].decode())["parse_errors"] == []
    assert runner.log_window_boundary(prefix + text) == len(prefix + text)


def test_port_probe_rejects_active_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with pytest.raises(runner.ValidationError, match=f"port {port}"):
            runner.check_port_available(port)


def test_port_probe_allows_reuse_after_connection_closes():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as client:
            accepted, _ = listener.accept()
            accepted.shutdown(socket.SHUT_RDWR)
            accepted.close()
            assert client.recv(1) == b""
    runner.check_port_available(port)


def test_old_timeline_cannot_prove_current_retrieval():
    old = successful_log().replace("22:42:13Z", "22:41:13Z")
    result = assess(old)
    assert result["status"] == "incomplete"
    assert "post-retrieval interval samples for both engines" in result["missing"]


def test_old_success_followed_by_current_empty_reports_is_not_current_proof():
    old = successful_log().replace("22:42:13Z", "22:41:13Z")
    current = sample(0, {"vllm:kvcr_state:('in_flight_ops',)": 0})
    current += sample(1, {"vllm:kvcr_state:('in_flight_ops',)": 0})
    result = assess(old + current)
    assert result["status"] == "incomplete"
    assert result["source_blocks"] is None


def test_wrong_block_counts_fail_at_deadline():
    result = assess(successful_log(blocks=4), final=True)
    assert result["status"] == "failed"
    assert any("cached blocks 5" in failure for failure in result["failures"])


def test_mismatched_bytes_fail_at_deadline():
    text = successful_log().replace(
        f"{runner.DESTINATION_BYTES}=1000", f"{runner.DESTINATION_BYTES}=999"
    )
    result = assess(text, final=True)
    assert result["status"] == "failed"
    assert any(
        "source bytes 1000 != destination bytes 999" in error
        for error in result["failures"]
    )


def test_partial_line_is_deferred_and_ansi_context_is_supported():
    text = successful_log()
    partial = text.rsplit("\n", 2)[0] + "\nunfinished"
    assert assess(partial)["status"] == "incomplete"
    colored = text.replace("Engine", "\x1b[32mEngine").replace(
        ": statistics", "\x1b[0m: statistics"
    )
    assert assess(colored)["status"] == "passed"


def response(rank, cached, worker=42):
    return {
        "http_status": 200,
        "error": None,
        "body": {
            "choices": [{"text": "arbitrary model continuation"}],
            "nvext": {
                "worker_id": {"decode_worker_id": worker, "decode_dp_rank": rank}
            },
            "usage": {"prompt_tokens_details": {"cached_tokens": cached}},
        },
    }


def test_http_identity_and_cache_geometry_are_required():
    assert runner.check_http_pair(response(0, 0), response(1, 320), 42)["passed"]
    wrong_rank = runner.check_http_pair(response(0, 0), response(0, 320), 42)
    assert not wrong_rank["passed"]
    assert any("wrong DP rank" in error for error in wrong_rank["failures"])


@pytest.mark.parametrize(
    "target", [response(1, 0), response(1, 319), response(1, 320, worker=99)]
)
def test_wrong_worker_or_miss_does_not_pass_http(target):
    assert not runner.check_http_pair(response(0, 0), target, 42)["passed"]


def test_artifact_sanitization_keeps_lines_and_redacts_signed_queries(tmp_path):
    path = tmp_path / "worker-window.log"
    path.write_text("download https://cdn.example/model?signature=secret\nmetric=3\n")
    runner.sanitize_logs(tmp_path)
    assert path.read_text() == (
        "download https://cdn.example/model?[REDACTED]\nmetric=3\n"
    )
    assert (tmp_path / "log-sanitization.json").is_file()


@pytest.mark.parametrize("status", ["passed", "failed", "running"])
def test_junit_only_reports_a_completed_pass_as_success(tmp_path, status):
    runner.write_junit(
        tmp_path,
        {
            "status": status,
            "started_at": "2026-09-18T00:00:00+00:00",
            "finished_at": "2026-09-18T00:00:02+00:00",
        },
    )
    suite = ET.parse(tmp_path / "junit.xml").getroot()
    assert suite.attrib["tests"] == "1"
    assert suite.attrib["time"] == "2.0"
    assert suite.attrib["failures"] == ("0" if status == "passed" else "1")
    assert (suite.find("testcase/failure") is None) == (status == "passed")


def test_cli_readiness_failure_returns_nonzero_and_emits_junit(tmp_path):
    output = tmp_path / "run"
    worker_log = tmp_path / "worker.log"
    worker_log.write_text("")
    # Reserve an unused local port without listening; no real service is touched.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        result = subprocess.run(
            [
                sys.executable,
                str(_PATH),
                "--output-dir",
                str(output),
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--worker-log",
                str(worker_log),
                "--timeout",
                "0.2",
                "--readiness-timeout",
                "0.1",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    assert result.returncode == 1, result.stdout + result.stderr
    evidence = json.loads((output / "result.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["peer_transfer_verified"] is False
    suite = ET.parse(output / "junit.xml").getroot()
    assert suite.attrib["failures"] == "1"
    assert suite.find("testcase/failure") is not None
    assert json.loads((output / "cleanup.json").read_text())["owned_processes"] == []
