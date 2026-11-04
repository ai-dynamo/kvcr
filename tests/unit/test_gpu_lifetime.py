# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from _gpu_test_utils import (
    DOMAIN,
    TLS,
    GPUAgent,
    GPUPinning,
    acknowledge_gpu_peer,
    gpu_envelope,
    gpu_hint,
    new_gpu,
    refs,
    regions,
    source_message,
)
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _decode_control_message,
    _decode_notif,
    _HintEnvelope,
    _new_kvcr,
    _op_entries,
    _poll_until,
    _record_hint_results,
    _use_nixl_agent,
    _wait_until,
    _write_done_notification,
)

from kvcr import KVCR, KVCRBindings
from kvcr.backends.remote_ops import (
    _StartWriteGPU,
    _TargetMetadata,
    _TargetPullOp,
    _TargetPullState,
)
from kvcr.config import KVCRBackendConfigs, KVCRConfig, RemoteFWGPUOptions
from kvcr.hints.results import HintStatus
from kvcr.types import BlockKey, CacheTier, MemoryRef, QueryStatus, RegionDescriptor


def test_legacy_cpu_hint_cannot_authorize_gpu_endpoint(monkeypatch):
    kvcr, agent, _, control = new_gpu(monkeypatch)
    results = _record_hint_results(monkeypatch, kvcr)
    hint = gpu_hint()
    hint["actions"][0]["action_type"] = "kv.fetch"
    kvcr.submit_hint(msgspec.convert(hint, type=_HintEnvelope), request_id="bad")
    assert len(results) == 1 and results[0].status is HintStatus.REJECTED
    assert not kvcr._core._remote_fw_dram._request_hints
    assert control.sent == [] and agent.xfers == []


@pytest.mark.parametrize("change", ["legacy", "domain", "transport", "host", "managed"])
def test_gpu_source_refuses_wrong_contract_and_host_peer_without_dma(
    monkeypatch, change
):
    kvcr, agent, pinning, control = new_gpu(monkeypatch, owner="source")
    payload = source_message()
    if change == "legacy":
        payload["type"] = "start_write"
    if change == "domain":
        payload["compatibility_domain"] = "other"
    if change == "transport":
        payload["transport"] = "cpu"
    if change == "host":
        payload["target_regions"][0]["main"] = msgspec.structs.replace(
            regions("target")[0], mem_type="DRAM"
        )
    if change == "managed":
        payload["target_regions"][1]["cpu"] = regions("target")[0]
    control.incoming.append(msgspec.msgpack.encode(payload))
    _wait_until(lambda: bool(agent.sent_notifs) or bool(control.sent))
    assert agent.xfers == pinning.searches == []
    assert agent.remote_agents == []


def test_cpu_source_never_executes_new_gpu_message():
    agent, control = FakeNixlAgent(), FakeBytesControl("tcp://source:1")
    kvcr = _new_kvcr(agent, FakePrimaryPinning(), control, name="source")
    control.incoming.append(msgspec.msgpack.encode(source_message()))
    _wait_until(lambda: bool(agent.sent_notifs) or bool(control.sent))
    assert agent.xfers == []
    assert _decode_notif(agent.sent_notifs[-1][1])["success"] is False
    list(kvcr.poll_completed())


def test_gpu_target_emits_distinct_wire_and_rejects_domain_and_storage(monkeypatch):
    kvcr, agent, _, control = new_gpu(monkeypatch)
    results = _record_hint_results(monkeypatch, kvcr)
    kvcr.submit_hint(gpu_envelope(domain="wrong"), request_id="bad")
    assert len(results) == 1 and results[0].status is HintStatus.REJECTED
    assert "domain" in results[0].reason
    assert kvcr.query([BlockKey(b"k0")], "bad") == [(QueryStatus.MISS, None)]
    assert control.sent == []
    kvcr.submit_hint(gpu_envelope(), request_id="req")
    assert kvcr.query([BlockKey(b"k0")], "req") == [
        (QueryStatus.FETCHABLE, CacheTier.REMOTE_G1)
    ]
    for call in (
        lambda: kvcr.deposit({b"k0": refs("target", 2)}),
        lambda: kvcr.fetch([b"k0"]),
    ):
        with pytest.raises(ValueError, match="no KVCR-managed"):
            call()
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "req")
    _wait_until(lambda: bool(control.sent))
    probe = _decode_control_message(control.sent[-1][1])
    assert probe["type"] == "target_metadata"
    assert not {
        "target_agent_metadata",
        "target_regions",
        "dst_descriptors",
    }.intersection(probe)
    acknowledge_gpu_peer(control, handle)
    message = _decode_control_message(control.sent[-1][1])
    assert message["type"] == "start_write_gpu"
    assert message["compatibility_domain"] == DOMAIN
    assert message["source_memory_kind"] == message["target_memory_kind"] == "gpu"
    assert list(kvcr.poll_completed()) == []
    agent.notifs["source"] = [_write_done_notification(handle)]
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({BlockKey(b"k0"): True}))]


def test_gpu_target_timeout_quarantines_until_actual_terminal(monkeypatch):
    errors = []
    kvcr, agent, _, control = new_gpu(monkeypatch, on_resilience_event=errors.append)
    clock = [0.0]
    kvcr._core._clock = lambda: clock[0]
    kvcr.submit_hint(gpu_envelope(), request_id="req")
    handle = kvcr.deliver({BlockKey(b"k0"): refs("target", 2)}, "req")
    _wait_until(lambda: bool(control.sent))
    acknowledge_gpu_peer(control, handle)
    clock[0] = 1.0
    _wait_until(
        lambda: any(
            _decode_control_message(raw)["type"] == "write_probe"
            for _, raw in control.sent
        )
    )
    assert list(kvcr.poll_completed()) == []
    clock[0] = 5.0
    _poll_until(kvcr, lambda _: bool(errors))
    assert list(kvcr.poll_completed()) == []
    assert [e.state for e in errors] == ["uncertain"]
    assert kvcr._core._block_record_map[BlockKey(b"k0")].active_op_ids
    agent.notifs["source"] = [_write_done_notification(handle)]
    assert _poll_until(kvcr, bool) == [(handle, _op_entries({BlockKey(b"k0"): False}))]
    assert [e.state for e in errors] == ["uncertain", "quiesced"]
    assert (
        list(kvcr.poll_completed()) == []
    )  # A late success never validates canceled data.


def test_gpu_partial_pin_preserves_original_target_key_position(monkeypatch):
    kvcr, agent, pinning, control = new_gpu(
        monkeypatch, owner="source", pinning=GPUPinning(missing=(0,))
    )
    agent.state = "DONE"
    control.incoming.append(msgspec.msgpack.encode(source_message()))
    _poll_until(kvcr, lambda _: bool(pinning.unpins))
    _, src, _, dst, _, notification = agent.xfers[0]
    assert src == [(1064, 16, 3), (2032, 8, 3)]
    assert dst == [(9480, 16, 5), (10240, 8, 5)]
    assert _decode_notif(notification)["completed_indices"] == [1]
    assert pinning.unpins == ["pin"]


def test_gpu_labelled_rank_banks_share_one_write_and_preserve_device_ids(monkeypatch):
    monkeypatch.setenv("UCX_TLS", TLS)
    monkeypatch.setenv("NIXL_UCX_CUDA_IPC_ONLY", "1")

    def rank_refs(owner, slot, *, reverse=False):
        labels = (":rank1", ":rank0") if reverse else (":rank0", ":rank1")
        return [
            MemoryRef(end_point_name=owner, element_index=slot, label=label)
            for label in labels
        ]

    def rank_regions(owner):
        source = owner == "source"
        return [
            RegionDescriptor(
                addr=address,
                size=16,
                stride=64 if source else 80,
                count=8,
                mem_type="VRAM",
                device_Id=device,
                label=f":rank{rank}",
            )
            for rank, (address, device) in enumerate(
                ((1000, 3), (2000, 7)) if source else ((9000, 5), (10000, 9))
            )
        ]

    class RankPinning(FakePrimaryPinning):
        def request_pin(self, keys):
            request = super().request_pin(keys)
            _, (_, mapping) = self.completed.pop()
            self.completed.append(
                (
                    request,
                    (
                        "pin",
                        {
                            key: rank_refs(self.agent_name, (5, 1)[i], reverse=True)
                            for i, key in enumerate(mapping)
                        },
                    ),
                )
            )
            return request

    agent, pinning = GPUAgent(), RankPinning()
    control = FakeBytesControl("tcp://source:1")
    kvcr = _new_kvcr(
        agent,
        pinning,
        control,
        KVCRConfig(nixl_agent_name="source", pool_layouts=[("", 16)]),
        name="source",
        framework_regions=rank_regions("source"),
        remote_gpu_options=RemoteFWGPUOptions(compatibility_domain=DOMAIN),
    )
    message = source_message()
    message["target_regions"] = (
        {region.label: region for region in rank_regions("target")}, {}
    )
    message["dst_descriptors"] = [rank_refs("target", slot) for slot in (2, 6)]
    agent.state = "DONE"
    control.incoming.append(msgspec.msgpack.encode(message))
    _poll_until(kvcr, lambda _: bool(pinning.unpins))
    assert len(agent.xfers) == 1
    _, src, _, dst, _, notification = agent.xfers[0]
    assert set(agent.prep_types) == {"VRAM"}
    assert src == [(1320, 16, 3), (2320, 16, 7), (1064, 16, 3), (2064, 16, 7)]
    assert dst == [(9160, 16, 5), (10160, 16, 9), (9480, 16, 5), (10480, 16, 9)]
    assert _decode_notif(notification)["completed_indices"] == [0, 1]
    assert len(pinning.searches) == 1 and pinning.unpins == ["pin"]


def test_gpu_source_timeout_keeps_allocation_pinned_until_native_done(monkeypatch):
    errors = []
    kvcr, agent, pinning, control = new_gpu(
        monkeypatch, owner="source", on_resilience_event=errors.append
    )
    clock = [0.0]
    kvcr._core._clock = lambda: clock[0]
    try:
        control.incoming.append(msgspec.msgpack.encode(source_message()))
        _poll_until(kvcr, lambda _: bool(agent.xfers))
        clock[0] = 1.0
        _wait_until(lambda: bool(agent.sent_notifs))
        assert agent.released_xfers == pinning.unpins == []
        clock[0] = 5.0
        _poll_until(kvcr, lambda _: bool(errors))
        assert [e.state for e in errors] == ["uncertain"]
        assert agent.released_xfers == pinning.unpins == []
        assert kvcr._core._framework_pin_keys
        agent.state = "DONE"
        _poll_until(kvcr, lambda _: bool(pinning.unpins))
        assert [e.state for e in errors] == ["uncertain", "quiesced"]
        assert agent.released_xfers == [1]
        assert pinning.unpins == ["pin"]
    finally:
        agent.state = "DONE"
        kvcr.close()


@pytest.mark.parametrize("memory_type", ["VRAM_SEG", "VRAM"])
def test_gpu_native_vram_enum_and_legacy_alias_are_supported(monkeypatch, memory_type):
    agent = GPUAgent()
    agent.get_backend_mem_types = lambda _: ["DRAM_SEG", memory_type]
    kvcr, _, _, _ = new_gpu(monkeypatch, agent=agent)
    assert agent.registrations
    assert {kind for _, kind in agent.registrations} == {"VRAM"}
    kvcr.close()


@pytest.mark.parametrize("memory_types", [["DRAM_SEG"], ["DRAM"], ["UNKNOWN"], []])
def test_gpu_native_backend_without_vram_support_fails_closed(
    monkeypatch, memory_types
):
    agent = GPUAgent()
    agent.get_backend_mem_types = lambda _: memory_types
    with pytest.raises(RuntimeError, match="native UCX VRAM"):
        new_gpu(monkeypatch, agent=agent)
    assert agent.xfers == agent.registrations == []


@pytest.mark.parametrize("value", [None, "0", "true", ""])
def test_gpu_requires_explicit_native_direct_policy_flag(monkeypatch, value):
    monkeypatch.setenv("UCX_TLS", TLS)
    if value is None:
        monkeypatch.delenv("NIXL_UCX_CUDA_IPC_ONLY", raising=False)
    else:
        monkeypatch.setenv("NIXL_UCX_CUDA_IPC_ONLY", value)
    agent = GPUAgent()
    with _use_nixl_agent(agent), pytest.raises(ValueError, match="CUDA_IPC_ONLY=1"):
        KVCR(
            KVCRConfig(
                nixl_agent_name="target", pool_layouts=[("main", 16), ("index", 8)]
            ),
            KVCRBindings(Mock(), Mock(), Mock(), framework_control=FakeBytesControl()),
            KVCRBackendConfigs(
                framework_regions=regions("target"),
                remote_fw_gpu=RemoteFWGPUOptions(compatibility_domain=DOMAIN),
            ),
        )
    assert agent.registrations == agent.xfers == []


def test_gpu_all_missing_source_keys_release_pin_without_native_write(monkeypatch):
    kvcr, agent, pinning, control = new_gpu(
        monkeypatch, owner="source", pinning=GPUPinning(missing=(0, 1))
    )
    control.incoming.append(msgspec.msgpack.encode(source_message()))
    _poll_until(
        kvcr,
        lambda _: bool(pinning.unpins)
        and bool(agent.sent_notifs)
        and kvcr._core._outstanding_operations == 0,
    )
    assert pinning.unpins == ["pin"]
    assert agent.xfers == agent.transfers == agent.released_xfers == []
    result = _decode_notif(agent.sent_notifs[-1][1])
    assert result["success"] is True and result["completed_indices"] == []
    assert not kvcr._core._framework_pin_keys
    assert not kvcr._core._block_record_map


@pytest.mark.parametrize("completed_indices", [(), (1,)])
def test_gpu_partial_target_completion_never_validates_missing_key(
    monkeypatch, completed_indices
):
    kvcr, agent, _, control = new_gpu(monkeypatch)
    kvcr.submit_hint(gpu_envelope(), request_id="partial")
    keys = [BlockKey(b"k0"), BlockKey(b"k1")]
    handle = kvcr.deliver(
        {keys[0]: refs("target", 2), keys[1]: refs("target", 6)}, "partial"
    )
    acknowledge_gpu_peer(control, handle)
    agent.notifs["source"] = [
        _write_done_notification(handle, completed_indices=completed_indices)
    ]
    assert _poll_until(kvcr, bool) == [
        (
            handle,
            _op_entries(
                {key: index in completed_indices for index, key in enumerate(keys)}
            ),
        )
    ]
    assert list(kvcr.poll_completed()) == []
    assert not kvcr._core._block_record_map
    assert agent.xfers == []


def gpu_pull_operation(*, send_result=True):
    clock = [0.0]
    backend = SimpleNamespace(
        _gpu=True,
        _kvcr=SimpleNamespace(_clock=lambda: clock[0]),
        _dangling_ops=SimpleNamespace(sources={}),
        _metadata_acked_sources=set(),
        _options=SimpleNamespace(metadata_retry_interval_ms=100),
        _send_control=Mock(return_value=send_result),
        _record_progress_duration=Mock(),
    )
    op = _TargetPullOp(
        op_id=("target", 7),
        keys=frozenset({BlockKey(b"k0")}),
        ordered_keys=(BlockKey(b"k0"),),
        started_at=0.0,
        deadline=1.0,
        state=_TargetPullState.START_WRITE,
        local_fill=False,
        remote_ctrl_ep="peer",
        _backend=backend,
    )
    return op, backend, clock


def test_gpu_capability_probe_backoff_cannot_offer_destination_early():
    op, backend, clock = gpu_pull_operation()
    assert op.progress(Mock(), None) == (False, True)
    assert isinstance(backend._send_control.call_args.args[2], _TargetMetadata)
    assert backend._send_control.call_args.args[2].gpu_capability_only
    clock[0] = 0.05
    assert op.progress(Mock(), None) == (False, False)
    assert backend._send_control.call_count == 1
    clock[0] = 0.1
    assert op.progress(Mock(), None) == (False, True)
    assert backend._send_control.call_count == 2
    assert all(
        isinstance(call.args[2], _TargetMetadata)
        for call in backend._send_control.call_args_list
    )
    op.gpu_peer_verified = True
    backend._dangling_ops.sources["peer"] = "verified"
    assert op.progress(Mock(), None) == (False, True)
    assert isinstance(backend._send_control.call_args.args[2], _StartWriteGPU)
    assert op.state is _TargetPullState.WAITING_WRITE_DONE


def test_gpu_failed_capability_probe_finishes_without_destination_offer():
    op, backend, _ = gpu_pull_operation(send_result=False)
    assert op.progress(Mock(), None) == (True, True)
    assert op.state is _TargetPullState.FINISHED and op.success is False
    assert backend._send_control.call_count == 1
    assert isinstance(backend._send_control.call_args.args[2], _TargetMetadata)
    assert backend._send_control.call_args.args[2].gpu_capability_only


@pytest.mark.parametrize("change", ["managed", "foreign", "out_of_bounds"])
def test_gpu_deliver_rejects_invalid_destinations_before_offering_memory(
    monkeypatch, change
):
    kvcr, agent, pinning, control = new_gpu(monkeypatch)
    kvcr.submit_hint(gpu_envelope(), request_id="invalid-destination")
    descriptors = refs("target", 2)
    if change == "managed":
        descriptors[0] = msgspec.structs.replace(descriptors[0], end_point_name="")
    elif change == "foreign":
        descriptors[0] = msgspec.structs.replace(
            descriptors[0], end_point_name="other-agent"
        )
    else:
        descriptors[0] = msgspec.structs.replace(descriptors[0], element_index=8)
    with pytest.raises(ValueError):
        kvcr.deliver({BlockKey(b"k0"): descriptors}, "invalid-destination")
    assert control.sent == agent.remote_agents == agent.xfers == pinning.searches == []
    assert not kvcr._core._block_record_map
