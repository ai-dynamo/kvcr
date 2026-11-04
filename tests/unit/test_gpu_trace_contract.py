# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic metadata guards; these fixtures are not CUDA execution evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "gpu_trace_contract", Path(__file__).parents[1] / "integration/read_gpu_trace.py"
)
trace_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace_module)


def packet(tmp_path, events):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    peer = tmp_path / "peer.json"
    peer.write_text(
        json.dumps(
            {
                "role": "source",
                "physical_device": {"uuid": "SOURCE_ONLY"},
                "profile": {
                    "requested": True,
                    "status": "TRACE_CAPTURED_UNREVIEWED",
                    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                },
            }
        )
    )
    return trace, peer


def test_empty_cuda_trace_is_never_positive_proof(tmp_path):
    trace, peer = packet(tmp_path, [{"name": "poll", "cat": "cpu_op"}])
    result = trace_module.inspect(trace, peer)
    assert result["empty_or_unpopulated_cuda_trace"] is True
    assert result["physical_no_host_staging_qualified"] is False
    assert result["status"].startswith("UNQUALIFIED")


def test_all_copy_api_and_activity_frames_keep_device_byte_metadata(tmp_path):
    events = [
        {"name": "cuMemcpyAsync", "cat": "cuda_driver", "args": {"correlation": 7}},
        {
            "name": "Memcpy DtoD",
            "cat": "gpu_memcpy",
            "args": {"bytes": 123863040, "srcDevice": 0, "dstDevice": 1},
        },
        {"name": "Memcpy HtoD", "cat": "gpu_memcpy", "args": {"bytes": 65536}},
    ]
    trace, peer = packet(tmp_path, events)
    result = trace_module.inspect(trace, peer)
    assert result["all_memcpy_api_and_activity_events"] == events
    assert result["cuda_activity_event_count"] == 2
    assert result["physical_no_host_staging_qualified"] is False


def test_trace_byte_mismatch_rejected(tmp_path):
    trace, peer = packet(tmp_path, [])
    trace.write_text('{"traceEvents":[]}\n')
    with pytest.raises(AssertionError):
        trace_module.inspect(trace, peer)
