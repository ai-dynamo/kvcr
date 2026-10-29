"""Instrumentation on the current typed protocol and native lifetime fences."""

import json
from types import SimpleNamespace

import msgspec
import pytest
from _kvcr_test_utils import _decode_notif, _start_write_message, _wait_until
from test_no_pinning_mode import _request
from test_no_pinning_mode import source_factory as source_factory

from kvcr.progress import _TransferRef


def modules():
    try:
        from kvcr import compact_telemetry, operation_capture
    except ImportError:
        pytest.fail("operation capture modules missing")
    return compact_telemetry, operation_capture


def test_immediate_done_and_release_retry_keep_exact_landmarks(tmp_path, monkeypatch):
    from test_progress import _mem, _transfer_progress, _TransferAgent

    telemetry, capture = modules()
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    agent = _TransferAgent()
    agent.transfer_result = "DONE"
    agent.release_failures = 1
    progress = _transfer_progress(agent)
    context = capture.Context(
        output,
        SimpleNamespace(
            _kvcr=SimpleNamespace(nixl_agent_name="source"),
            _dangling_ops=SimpleNamespace(incarnation="inc"),
            _options=SimpleNamespace(no_pinning_mode=True),
        ),
        "source",
        (b"key",),
    )
    monkeypatch.setattr(capture._local, "context", context, raising=False)
    transfer_id, submitted = progress.submit_transfer(
        "WRITE",
        [_mem(0)],
        [_mem(1, owner="remote-agent")],
        remote_side_agent="remote-agent",
    )
    assert submitted
    assert "first_observed_done_ns" in context.timings
    done = context.timings["first_observed_done_ns"]
    assert progress.poll_transfer(transfer_id) is None
    assert "native_handle_released_ns" not in context.timings
    assert transfer_id in progress._active_transfers
    assert progress.poll_transfer(transfer_id)[0]
    assert context.timings["first_observed_done_ns"] == done
    assert done <= context.timings["native_handle_released_ns"]
    output.close()


def test_typed_source_request_is_captured_without_dict_payload(tmp_path, monkeypatch):
    from kvcr.remote_ops import _StartWrite

    telemetry, capture = modules()
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    backend = SimpleNamespace(
        _kvcr=SimpleNamespace(nixl_agent_name="source"),
        _dangling_ops=SimpleNamespace(incarnation="inc"),
        _options=SimpleNamespace(no_pinning_mode=True),
    )

    @capture.request("source")
    def handle(self, progress, payload):
        assert capture.current().keys == (b"exact",)
        capture.reason("source_stalled")

    handle(
        backend,
        None,
        _StartWrite(
            op_handle=1,
            remaining_timeout_ms=10,
            keys=(b"exact",),
            dst_descriptors=((),),
        ),
    )
    output.close()
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    terminal = next(row for row in rows if row["event"] == "source_nixl_terminal")
    assert terminal["no_pinning_mode"] is True
    assert terminal["reason"] == "source_stalled"
    assert telemetry.validate_capture(tmp_path)["valid"]


def test_progress_hook_captures_real_idle_iteration(tmp_path, monkeypatch):
    from test_progress import _transfer_progress, _TransferAgent

    telemetry, _ = modules()
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    progress = _transfer_progress(_TransferAgent())
    assert progress._run_one_iteration() is False
    output.close()
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    assert (
        next(row for row in rows if row["event"] == "progress_schedule")["idle_turns"]
        == 1
    )


def test_source_capture_has_claim_and_cleanup_timing(
    source_factory, tmp_path, monkeypatch, caplog
):
    telemetry, _ = modules()
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    caplog.set_level("INFO", logger="kvcr.core")
    source, agent, control, pinning, _ = source_factory()
    try:
        before = len(agent.xfers)
        _request(control, [b"missing", b"a"])
        _wait_until(lambda: len(agent.xfers) == before + 1)
        _wait_until(lambda: source._core._local_dram._export_readers() == 0)
        source.query([b"a", b"missing"])
    finally:
        source.close()
        output.close()
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    terminal = next(row for row in rows if row["event"] == "source_nixl_terminal")
    assert terminal["status"] == "partial" and terminal["completed_indices"] == [1]
    clock = terminal["timing_ns"]
    assert (
        clock["source_receive_ns"]
        <= clock["claim_begin_ns"]
        <= clock["claim_end_ns"]
        <= clock["submit_begin_ns"]
    )
    assert (
        clock["first_observed_done_ns"]
        <= clock["native_handle_released_ns"]
        <= clock["cleanup_done_ns"]
        <= clock["source_end_ns"]
    )
    assert not pinning.searches
    assert not [
        record
        for record in caplog.records
        if "no_pinning_mode event=" in record.getMessage()
    ]
    assert (
        next(row for row in rows if row["event"] == "query_outcomes")["denominator"]
        == 2
    )
    assert telemetry.validate_capture(tmp_path, ["source"])["valid"]


@pytest.mark.parametrize("keys", [None, 17, [None], b"not-a-key-sequence"])
def test_malformed_typed_ingress_has_failure_capture(
    source_factory, tmp_path, monkeypatch, keys
):
    import msgspec
    from _kvcr_test_utils import _start_write_message

    telemetry, _ = modules()
    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    source, agent, control, _, _ = source_factory()
    received = []
    monkeypatch.setattr(
        source._core._remote_fw_dram,
        "_notify_start_write_failure",
        lambda progress, payload, handle: received.append(handle),
    )
    before = len(agent.xfers)
    payload = msgspec.msgpack.decode(
        _start_write_message(999, b"a", target_agent="target")
    )
    payload["keys"] = keys
    try:
        control.incoming.append(msgspec.msgpack.encode(payload))
        _wait_until(lambda: received == [999])
        assert len(agent.xfers) == before
    finally:
        source.close()
        output.close()
    terminal = next(
        row
        for row in map(json.loads, output.path.read_text().splitlines())
        if row["event"] == "source_nixl_terminal"
    )
    assert terminal["reason"] == "malformed_start_write"
    assert telemetry.validate_capture(tmp_path)["valid"]


def test_compact_target_mapping_preserves_exact_keys_and_partial_denominator(
    tmp_path, monkeypatch
):
    from test_kvcr_remote_target import (
        BlockKey,
        CacheTier,
        FakeBytesControl,
        FakeNixlAgent,
        FakePrimaryPinning,
        QueryStatus,
        RemoteFWDramOptions,
        _ConstantHashAdapter,
        _decode_control_message,
        _mem_descriptor,
        _new_kvcr,
        _op_entries,
        _poll_until,
        _router_hint,
        _write_done_notification,
    )

    from kvcr import compact_telemetry as telemetry

    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    agent, control = FakeNixlAgent(), FakeBytesControl("tcp://target:1")
    target = _new_kvcr(
        agent,
        FakePrimaryPinning(),
        control,
        key_adapter=_ConstantHashAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )
    keys = (BlockKey(b"exact-target-key-0"), BlockKey(b"exact-target-key-1"))
    try:
        target.submit_hint(_router_hint("tcp://source:1"), request_id="req")
        assert (
            target.query(keys, "req")
            == [(QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)] * 2
        )
        handle = target.deliver(
            {key: [_mem_descriptor(index)] for index, key in enumerate(keys)},
            request_id="req",
        )
        _wait_until(lambda: bool(control.sent))
        message = _decode_control_message(control.sent[0][1])
        assert message["keys"] == list(keys)
        agent.notifs["source"] = [
            _write_done_notification(handle, completed_indices=(0,))
        ]
        assert _poll_until(target, bool) == [
            (handle, _op_entries({keys[0]: True, keys[1]: False}))
        ]
    finally:
        target.close()
        output.close()
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    terminal = next(row for row in rows if row["event"] == "target_terminal")
    assert terminal["requested_blocks"] == 2 and terminal["blocks"] == 1
    assert terminal["status"] == "partial" and terminal["bytes"] == 16
    journal = list(telemetry.iter_key_journal(output.key_path))
    get = next(row for row in journal if row["event"] == "target_get")
    assert (
        get["keys"] == list(keys) and get["operation_key"] == terminal["operation_key"]
    )
    assert telemetry.validate_capture(tmp_path, ["target"])["valid"]


def test_guard_validates_only_claimed_subset_after_claim(source_factory, monkeypatch):
    source, agent, control, pinning, _ = source_factory()
    core = source._core
    validate = core._validate_descriptors
    seen = []

    def checked(descriptors, **kwargs):
        seen.append((core._local_dram._export_readers(), tuple(descriptors)))
        return validate(descriptors, **kwargs)

    monkeypatch.setattr(core, "_validate_descriptors", checked)
    payload = msgspec.msgpack.decode(
        _start_write_message(901, b"a", target_agent="target")
    )
    payload.update(
        keys=[b"missing", b"a"],
        dst_descriptors=[[_TransferRef("target", 1000)], [_TransferRef("target", 1)]],
    )
    before = len(agent.xfers)
    try:
        control.incoming.append(msgspec.msgpack.encode(payload))
        _wait_until(lambda: bool(seen))
        assert seen == [(1, (_TransferRef("target", 1),))]
        _wait_until(lambda: len(agent.xfers) == before + 1)
        assert _decode_notif(agent.xfers[-1][5])["completed_indices"] == [1]
        assert not pinning.searches
    finally:
        source.close()


def test_selected_descriptor_failure_releases_claim_before_submission(
    source_factory, monkeypatch
):
    source, agent, control, pinning, _ = source_factory()
    core = source._core
    observed = []

    def invalid(*args, **kwargs):
        observed.append(core._local_dram._export_readers())
        raise ValueError("selected destination rejected")

    monkeypatch.setattr(core, "_validate_descriptors", invalid)
    before = len(agent.xfers)
    try:
        _request(control, [b"a"], handle=902)
        _wait_until(lambda: bool(observed))
        assert observed == [1]
        _wait_until(lambda: core._local_dram._export_readers() == 0)
        assert len(agent.xfers) == before
        assert not core._remote_fw_dram._dangling_ops.source_writes
        assert not pinning.searches
    finally:
        source.close()


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("malformed", "malformed_start_write"),
        ("peer", "peer_metadata_validation_failed"),
        ("native", "native_state:ERR"),
        ("deadline", "source_deadline_expired"),
    ],
)
def test_compact_source_failure_reason_is_explicit(
    source_factory, tmp_path, monkeypatch, failure, expected
):
    import json

    from kvcr import compact_telemetry as telemetry

    output = telemetry.BufferedWriter(
        tmp_path, source_archive_sha256="test", start_thread=False
    )
    monkeypatch.setattr(telemetry, "writer", lambda: output)
    source, agent, control, _, _ = source_factory()
    backend = source._core._remote_fw_dram
    now = [0.0]
    source._core._clock = lambda: now[0]
    try:
        if failure == "peer":

            def bad_peer(*args, **kwargs):
                raise ValueError("bad peer")

            monkeypatch.setattr(backend, "_remote_agent", bad_peer)
        before = len(agent.xfers)
        if failure in ("native", "deadline"):
            agent.state = "PROC"
        if failure == "malformed":
            _request(control, [b"a"], handle=904, remaining_timeout_ms=-1)
        else:
            _request(control, [b"a"], handle=904)
        _wait_until(lambda: not control.incoming)
        if failure in ("native", "deadline"):
            _wait_until(lambda: len(agent.xfers) == before + 1)
            if failure == "native":
                agent.state = "ERR"
                _wait_until(
                    lambda: any(
                        state.outcome is False
                        for state in source._core._progress._active_transfers.values()
                    )
                )
            else:
                now[0] = 100.0
                _wait_until(
                    lambda: any(
                        status.abandoned
                        for status in backend._dangling_ops.source_writes.values()
                    )
                )
            assert source._core._local_dram._export_readers() == 1
            agent.state = "DONE"
            _wait_until(lambda: source._core._local_dram._export_readers() == 0)
    finally:
        agent.state = "DONE"
        source.close()
        output.close()
    terminal = next(
        row
        for row in map(json.loads, output.path.read_text().splitlines())
        if row["event"] == "source_nixl_terminal"
    )
    assert terminal["result"] == "failed"
    assert any(expected in reason for reason in terminal["reasons"])
    assert telemetry.validate_capture(tmp_path, ["source"])["valid"]


@pytest.mark.parametrize("metrics_enabled", [False, True])
def test_source_core_boundary_reuses_metric_sample(monkeypatch, metrics_enabled):
    from kvcr.remote_fw_dram import _RemoteFWDram

    _, capture = modules()
    context = SimpleNamespace(side="source", timings={})
    backend = SimpleNamespace(_telemetry_enabled=metrics_enabled, _progress_metrics=[])
    monkeypatch.setattr(capture._local, "context", context, raising=False)
    samples = []

    def sampled():
        samples.append(1)
        return 12.5

    monkeypatch.setattr("kvcr.remote_fw_dram.time.monotonic", sampled)
    _RemoteFWDram._record_progress_duration(backend, "source_write", 10.0, "success")
    assert samples == [1]
    assert context.timings["core_source_write_ns"] == 12_500_000_000
    assert bool(backend._progress_metrics) is metrics_enabled
    if metrics_enabled:
        assert backend._progress_metrics[0][2] == 2.5
