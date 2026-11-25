# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Precise source-only transfer samples retain strict request boundaries."""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))

from phase1 import stack, telemetry  # noqa: E402

PREFIX = "CI_LOG_EPOCH=%(created).9f "


def formatted_sample(epoch, blocks):
    """Use the real stdlib Formatter/LogRecord expansion used by vLLM."""
    key = telemetry.metric(telemetry.TRANSFER_BLOCKS, telemetry.SOURCE_WRITE)
    record = logging.LogRecord(
        "vllm",
        logging.INFO,
        "metrics.py",
        108,
        f"KV Transfer metrics: {key}={blocks}",
        (),
        None,
    )
    record.created = epoch
    formatter = logging.Formatter(
        PREFIX + "(MainProcess pid=%(process)d) %(levelname)s "
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    return formatter.format(record) + "\n"


def test_source_only_precise_samples_obey_same_second_boundaries():
    start = datetime(2026, 11, 25, 10, 25, 9, 355000, tzinfo=timezone.utc)
    before, after = start.timestamp() - 0.1, start.timestamp() + 0.1
    text = (
        "2026-11-25T10:24:44.555623Z INFO previous seed HTTP request\n"
        + formatted_sample(before, 7)
        + formatted_sample(after, 36)
    )
    samples, errors = telemetry.parse_log(text)
    assert errors == []
    assert len(samples) == 2
    assert samples[0].timestamp == datetime.fromtimestamp(before, timezone.utc)
    assert samples[1].timestamp == datetime.fromtimestamp(after, timezone.utc)
    assert all(not sample.second_precision for sample in samples)
    assert samples[0].timestamp < start < samples[1].timestamp
    observed = telemetry.window(samples, errors, start)
    assert observed.blocks(telemetry.SOURCE_WRITE) == 36
    assert observed.samples == 1


@pytest.mark.parametrize(
    "epoch",
    [
        "nan",
        "inf",
        "-inf",
        "1e400",
        "-1.000000000",
        "253402300800.000000000",
        "bad",
        "1795602309",
    ],
)
def test_malformed_epoch_marker_cannot_fall_back_to_coarse_timestamp(epoch):
    key = telemetry.metric(telemetry.TRANSFER_BLOCKS, telemetry.SOURCE_WRITE)
    line = (
        f"CI_LOG_EPOCH={epoch} (MainProcess pid=1) INFO 11-25 10:25:09 "
        f"[metrics.py:108] KV Transfer metrics: {key}=36\n"
    )
    samples, errors = telemetry.parse_log(line)
    assert samples == []
    assert errors and "timestamp" in errors[0]
    assert telemetry.line_timestamp(line) is None


def test_epoch_marker_is_anchored_to_log_prefix():
    assert telemetry.precise_stamp("noise CI_LOG_EPOCH=1795602309.455000000") is None


@pytest.mark.parametrize("engine", [stack.VLLM, stack.SGLANG])
@pytest.mark.parametrize("topology", ["agg", "disagg"])
def test_actual_worker_launch_owns_precise_vllm_prefix_only(
    tmp_path, monkeypatch, engine, topology
):
    subject = stack.Stack(
        stack.Settings(
            model="m",
            output=tmp_path,
            engine=engine,
            mode=stack.NON_GUARD,
            topology=topology,
            gpus=(0, 0, 1),
            worker_env={"VLLM_LOGGING_PREFIX": "user-override "},
        ),
        tmp_path / "stack",
    )
    try:
        launches = {}

        def spawn(name, command, env):
            launches[name] = dict(env)
            return SimpleNamespace(name=name)

        monkeypatch.setattr(subject, "_spawn", spawn)
        monkeypatch.setattr(subject, "advertise_host", lambda: "127.0.0.1")
        for index, enabled, tag in [
            (0, False, "-reference"),
            (0, True, ""),
            (1, True, ""),
            (0, True, "-replacement"),
        ]:
            subject.start_worker(index, kvcr=enabled, tag=tag)
        if topology == "disagg":
            for enabled, tag in [(False, "-reference"), (True, "")]:
                subject.start_worker(2, kvcr=enabled, tag=tag)
        expected = PREFIX if engine == stack.VLLM else "user-override "
        assert all(env["VLLM_LOGGING_PREFIX"] == expected for env in launches.values())
    finally:
        subject.cleanup()


@pytest.mark.parametrize("topology", ["agg", "disagg"])
@pytest.mark.parametrize("real_kvcr_activity", [False, True])
def test_reference_phase_keeps_independence_guard_with_precise_logs(
    tmp_path, monkeypatch, topology, real_kvcr_activity
):
    """Exercise the real reference phase with actual launch env and Client I/O.

    Only process creation/readiness and the HTTP peer are substituted; the
    reference's log scan, bypass check, token comparison and evidence writes run.
    """
    import importlib.util
    import io
    import json

    from phase1 import client as client_module

    spec = importlib.util.spec_from_file_location(
        "precision_reference_runner", Path(__file__).parents[1] / "e2e/run_phase1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.arguments(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--mode",
            "nonguard",
            "--topology",
            topology,
            "--gpus",
            "0,0,1" if topology == "disagg" else "0,1",
            "--gpu-memory-utilization",
            "0.35",
            "--cases",
            "TC-01",
        ]
    )
    subject = module.Runner(args)
    launched = {}

    def spawn(name, command, env):
        prefix = env.get("VLLM_LOGGING_PREFIX", "")
        record = logging.LogRecord(
            "vllm",
            logging.INFO,
            "metrics.py",
            108,
            "Engine initialized with prefix cache",
            (),
            None,
        )
        record.created = 1795602309.455
        text = (
            logging.Formatter(prefix + "%(levelname)s %(message)s").format(record)
            + "\n"
        )
        if real_kvcr_activity:
            record.msg = "KVCR secondary tier initialized"
            text += (
                logging.Formatter(prefix + "%(levelname)s %(message)s").format(record)
                + "\n"
            )
        worker = SimpleNamespace(name=name, log_text=lambda: text)
        launched[name] = worker
        (subject.stack.evidence / f"{name}.launch.json").write_text(
            json.dumps({"command": command, "environment_overrides": env})
        )
        return worker

    def response(request, timeout):
        # A stable real Client reply from the substituted HTTP peer.
        result = io.BytesIO(
            json.dumps(
                {
                    "choices": [{"finish_reason": "length"}],
                    "nvext": {
                        "prompt_token_ids": [1, 2],
                        "completion_token_ids": [3, 4],
                    },
                    "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
                }
            ).encode()
        )
        result.status = 200
        result.headers = {"Content-Type": "application/json"}
        return result

    monkeypatch.setattr(subject.stack, "_spawn", spawn)
    monkeypatch.setattr(subject.stack, "wait_model", lambda client: None)
    monkeypatch.setattr(
        subject.stack,
        "discover_worker",
        lambda client, index: subject.stack.worker_ids.update(
            {index: str(index + 101)}
        ),
    )
    monkeypatch.setattr(subject.stack, "stop", lambda *children, **kwargs: None)
    monkeypatch.setattr(client_module, "urlopen", response)
    try:
        if real_kvcr_activity:
            with pytest.raises(module.ValidationError, match="not verified KVCR-free"):
                subject.reference_phase()
            assert subject.result["reference"]["kvcr_mentions_in_worker_log"] > 0
        else:
            subject.reference_phase()
            assert subject.result["reference"]["kvcr_mentions_in_worker_log"] == 0
        assert subject.result["reference"]["kvcr_disabled_in_launch"] is True
        assert subject.result["reference"]["prompts"]["tc01"]["stable"] is True
        assert len(list((subject.output / "reference").glob("*.request.json"))) == 2
        assert len(list((subject.output / "reference").glob("*.response.json"))) == 2
        assert set(launched) == {"reference-frontend", "worker-0-reference"} | (
            {"worker-2-reference"} if topology == "disagg" else set()
        )
    finally:
        subject.stack.cleanup()
