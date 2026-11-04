# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec
import pytest
from _gpu_test_utils import DOMAIN, gpu_envelope, gpu_hint, parse_gpu_hint
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _HintEnvelope,
    _new_kvcr,
    _record_hint_results,
)

from kvcr.hints.fetch import KV_FETCH
from kvcr.hints.fetch_gpu import KV_FETCH_GPU, _KVGPUFetchHint
from kvcr.hints.parser import parse_hints
from kvcr.hints.results import HintStatus


def test_gpu_hint_is_strict_and_isolated_from_cpu():
    hints, rejected = parse_gpu_hint(gpu_hint())
    assert rejected == []
    (parsed,) = hints
    assert isinstance(parsed, _KVGPUFetchHint)
    assert (
        parsed.source_endpoint,
        parsed.block_hashes,
        parsed.compatibility_domain,
    ) == ("tcp://source:1", frozenset({123}), DOMAIN)
    hints, rejected = parse_hints(gpu_envelope(), {KV_FETCH.action_type: KV_FETCH})
    assert hints == [] and rejected[0].reason == "unsupported_action_type"
    legacy = gpu_hint()
    legacy["actions"][0]["action_type"] = "kv.fetch"
    hints, rejected = parse_gpu_hint(legacy)
    assert hints == [] and rejected[0].reason == "unsupported_action_type"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_memory_kind", "cpu"),
        ("target_memory_kind", "cpu"),
        ("transport", "nixl"),
        ("compatibility_domain", ""),
        ("block_hashes", [True]),
        ("source_control_endpoint", ""),
    ],
)
def test_gpu_hint_rejects_unsafe_payload(field, value):
    hint = gpu_hint()
    hint["actions"][0]["payload"][field] = value
    hints, rejected = parse_gpu_hint(hint)
    assert hints == [] and rejected[0].status is HintStatus.REJECTED


@pytest.mark.parametrize(
    "field,value", [("protocol_version", "9"), ("action_version", "9")]
)
def test_gpu_hint_rejects_unknown_version(field, value):
    hint = gpu_hint()
    (hint if field == "protocol_version" else hint["actions"][0])[field] = value
    hints, rejected = parse_gpu_hint(hint)
    assert hints == [] and rejected[0].reason == "unsupported_version"


def test_gpu_hint_strict_version_does_not_change_cpu_version_tolerance():
    hint = gpu_hint()
    hint["protocol_version"] = "future"
    cpu = {**hint["actions"][0], "action_id": "cpu", "action_type": "kv.fetch"}
    hint["actions"].append(cpu)
    hints, rejected = parse_hints(
        msgspec.convert(hint, type=_HintEnvelope),
        {kind.action_type: kind for kind in (KV_FETCH, KV_FETCH_GPU)},
    )
    assert [type(parsed).ACTION_TYPE for parsed in hints] == ["kv.fetch"]
    assert [(row.action_id, row.reason) for row in rejected] == [
        ("gpu-fetch", "unsupported_version")
    ]


def test_gpu_hint_cpu_endpoint_rejects_before_storing_or_connecting(monkeypatch):
    agent, control = FakeNixlAgent(), FakeBytesControl("tcp://target:1")
    kvcr = _new_kvcr(agent, FakePrimaryPinning(), control, name="target")
    results = _record_hint_results(monkeypatch, kvcr)
    kvcr.submit_hint(gpu_envelope(), request_id="bad")
    assert len(results) == 1 and results[0].status is HintStatus.REJECTED
    assert not kvcr._core._remote_fw_dram._request_hints
    assert control.sent == [] and agent.xfers == []
