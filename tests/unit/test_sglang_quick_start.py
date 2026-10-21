# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the example's fail-closed response/telemetry checks."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sglang_quick_start",
    Path(__file__).parents[2] / "examples/sglang-hicache/verify.py",
)
QS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QS)


def events():
    return [
        {
            "choices": [{"finish_reason": "length"}],
            "nvext": {
                "completion_token_ids": list(range(16)),
                "prompt_token_ids": [10, 11],
                "worker_id": {"prefill_worker_id": 42, "decode_worker_id": 42},
                "engine_data": {"reused_tokens": 0},
            },
            "usage": {"completion_tokens": 16, "prompt_tokens": 2},
        }
    ]


def test_exact_deltas_and_worker():
    response = events()
    first = copy.deepcopy(response[0])
    first["choices"] = [{"finish_reason": None}]
    first.pop("usage")
    first["nvext"]["completion_token_ids"] = list(range(8))
    response[0]["nvext"]["completion_token_ids"] = list(range(8, 16))
    result = QS.grade_response([first, *response], [10, 11], 42)
    assert result["output_ids"] == list(range(16))
    assert result["cached_tokens"] == 0


@pytest.mark.parametrize(
    "case",
    [
        "wrong_worker",
        "no_worker",
        "no_cache",
        "no_usage",
        "short_output",
        "wrong_prompt",
        "duplicate_terminal",
        "error",
        "wrong_count",
        "bad_token_type",
    ],
)
def test_reject_invalid_completion(case):
    response = events()
    event = response[0]
    if case == "wrong_worker":
        event["nvext"]["worker_id"]["decode_worker_id"] = 43
    elif case == "no_worker":
        del event["nvext"]["worker_id"]
    elif case == "no_cache":
        del event["nvext"]["engine_data"]
    elif case == "no_usage":
        del event["usage"]
    elif case == "short_output":
        event["nvext"]["completion_token_ids"].pop()
    elif case == "wrong_prompt":
        event["nvext"]["prompt_token_ids"] = [11, 10]
    elif case == "duplicate_terminal":
        response.append(copy.deepcopy(event))
    elif case == "error":
        event["error"] = "failed"
    elif case == "wrong_count":
        event["usage"]["completion_tokens"] = 15
    elif case == "bad_token_type":
        event["nvext"]["completion_token_ids"][0] = True
    with pytest.raises(RuntimeError):
        QS.grade_response(response, [10, 11], 42)


def test_missing_cache_never_defaults_to_zero():
    response = events()
    del response[0]["nvext"]["engine_data"]
    response[0]["usage"]["prompt_tokens_details"] = {"cached_tokens": 0}
    assert QS.grade_response(response, [10, 11], 42)["cached_tokens"] == 0


def test_native_intervals_and_loss():
    metrics = {
        json.dumps(["counter", "kvcr_transfer_bytes", ["source_write"], "value"]): 123,
        json.dumps(["counter", "kvcr_transfer_blocks", ["source_write"], "value"]): 2,
        json.dumps(
            [
                "histogram",
                "kvcr_duration_seconds",
                ["transfer", "source_write", "success"],
                "count",
            ]
        ): 1,
    }
    line = "KVCR L3 native stats " + json.dumps({"metrics": metrics}) + "\n"
    assert QS.native_stats(line * 2) == {
        "bytes": 246,
        "blocks": 4,
        "success": 2,
        "failed": 0,
    }
    with pytest.raises(RuntimeError, match="dropped"):
        QS.native_stats('KVCR L3 native stats {"metrics":{"dropped_observations":1}}')


def test_save_never_overwrites(tmp_path):
    path = tmp_path / "receipt.json"
    QS.save(path, {"failed_attempt": True})
    with pytest.raises(FileExistsError):
        QS.save(path, {"passed": True})


@pytest.mark.parametrize("discovered", [False, True])
def test_no_generation_before_exact_router_readiness(tmp_path, monkeypatch, discovered):
    session = QS.Session(tmp_path, 24000, 1)
    worker = 42 if discovered else 43
    (tmp_path / "frontend.log").write_text(
        f"Adding worker WorkerWithDpRank {{ worker_id: {worker}, dp_rank: 0 }}\n"
    )
    monkeypatch.setattr(
        session, "wait", lambda check, timeout, why: QS.require(check(), why)
    )
    monkeypatch.setattr(QS, "get_json", lambda url: {"ready": False})
    monkeypatch.setattr(
        QS.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail(
            "Generation must not be used as a readiness probe"
        ),
    )
    with pytest.raises(RuntimeError, match="not ready" if discovered else "discovery"):
        session.request(42, [10, 11], "cold-control")
    assert not (tmp_path / "cold-control-request.json").exists()
