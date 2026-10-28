"""Merged upstream tracing must preserve Guard-only ownership and capture."""
import json
import threading

import pytest
from _kvcr_test_utils import _wait_until
from _nvtx_test_utils import payloads, spans
from test_guard_only_sources import _source, _request

from kvcr import _nvtx, compact_telemetry as telemetry


@pytest.mark.parametrize('level', ['off', 'low', 'medium'])
def test_guard_subset_nvtx_preserves_quiescence_and_compact(recording, monkeypatch, tmp_path, level):
    monkeypatch.setenv('KVCR_NVTX_LEVEL', level)
    if level == 'off':
        monkeypatch.setattr(_nvtx, '_load_backend', lambda: pytest.fail('off must not load optional backend'))
    output = telemetry.BufferedWriter(tmp_path, source_archive_sha256='test', start_thread=False)
    monkeypatch.setattr(telemetry, 'writer', lambda: output)
    source, agent, control, pinning, _ = _source(slots=1)
    allow_release = threading.Event()
    release = agent.release_xfer_handle
    agent.release_xfer_handle = lambda handle: release(handle) if allow_release.is_set() else False
    try:
        before = len(agent.xfers)
        agent.state = 'PROC'
        _request(control, [b'missing', b'a'], handle=990, sender_incarnation='target-inc')
        _wait_until(lambda: len(agent.xfers) == before + 1)
        op = next(op for op in source._core._progress._in_flight_ops.values() if getattr(op, 'guard_only', False))
        assert (op.nvtx_state is None) == (level == 'off')
        if level != 'off':
            assert op.nvtx_state.target_incarnation == 'target-inc'
            assert op.nvtx_state.requested_blocks == 2
            assert op.nvtx_state.trace is not None
        agent.state = 'DONE'
        _wait_until(lambda: 'first_observed_done_ns' in op._capture.timings)
        assert source._core._local_dram._export_readers() == 1
        assert 'native_handle_released_ns' not in op._capture.timings
        allow_release.set()
        _wait_until(lambda: source._core._local_dram._export_readers() == 0)
    finally:
        agent.state = 'DONE'
        allow_release.set()
        source.close()
        output.close()
    assert not pinning.searches
    terminal = next(json.loads(line) for line in output.path.read_text().splitlines()
                    if json.loads(line)['event'] == 'source_nixl_terminal')
    assert terminal['result'] == 'success' and terminal['status'] == 'partial'
    assert terminal['guard_only_sources'] and terminal['has_framework_pins'] is False
    assert terminal['initial_caller_handoff'] == 'absent_by_guard_only_progress_path'
    clock = terminal['timing_ns']
    assert clock['first_observed_done_ns'] <= clock['native_handle_released_ns'] <= clock['core_source_write_ns'] <= clock['cleanup_done_ns']
    assert telemetry.validate_capture(tmp_path, ['source'])['valid']
    assert not payloads(recording, 'source.pin.context')
    assert len(payloads(recording, 'source.write.completed')) == (level != 'off')
    assert len(spans(recording, 'nixl.write', 'end')) == (level != 'off')
    assert not recording.ranges and not recording.stack
