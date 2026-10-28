"""Generic native observations must never weaken handle lifetime fencing."""
from types import SimpleNamespace

import pytest

from kvcr.types import RegionDescriptor
from kvcr import operation_capture
from test_progress import _TransferAgent, _mem, _transfer_progress


def setup_capture():
    agent = _TransferAgent()
    agent.get_xfer_telemetry = lambda handle: SimpleNamespace(totalBytes=128, xferDuration=2000)
    progress = _transfer_progress(agent, prepare=False)
    regions = ({"": RegionDescriptor(addr=128, size=128, count=4),
                "small": RegionDescriptor(addr=1024, size=64, count=2, mem_type="VRAM", label="small")}, {})
    for name in ("", agent.name, "remote-agent"):
        progress.prepare_memory(name, regions)
    rows = []
    progress._capture_writer = SimpleNamespace(emit=lambda row: rows.append(row), invalidate=lambda why: pytest.fail(why))
    return agent, progress, rows


def submit(progress):
    return progress.submit_transfer('WRITE', (_mem(0, label='small'), _mem(1, label='small')),
                                    (_mem(0, owner='remote-agent'),),
                                    remote_side_agent='remote-agent', backend='UCX')[0]


def test_generic_capture_after_release_retry_only():
    agent, progress, rows = setup_capture()
    transfer = submit(progress)
    agent.release_failures = 1
    assert progress.poll_transfer(transfer, require_completion=True) is None
    assert rows == [] and transfer in progress._active_transfers
    assert progress.poll_transfer(transfer, require_completion=True)[0] is True
    assert transfer not in progress._active_transfers
    assert len(rows) == 1
    row = rows[0]
    assert row['event'] == 'native_transfer_terminal'
    assert row['source_mem_type'] == 'VRAM' and row['destination_mem_type'] == 'DRAM'
    assert row['source_agent'] == agent.name and row['target_agent'] == 'remote-agent'
    assert row['bytes'] == row['descriptor_bytes'] == 128
    assert row['native_seconds'] == .002
    assert row['host_submit_to_release_seconds'] > 0
    assert row['native_handle_released'] is True
    assert 'release_retry' in row['reasons']


@pytest.mark.parametrize('state', ['PROC', 'PEND', 'ERR'])
def test_generic_capture_no_early_release(state):
    agent, progress, rows = setup_capture()
    transfer = submit(progress)
    agent.state = state
    assert progress.poll_transfer(transfer, require_completion=True) is None
    assert rows == [] and transfer in progress._active_transfers
    assert not any(e.startswith('release-transfer:') for e in agent.events)
    agent.state = 'DONE'
    result = progress.poll_transfer(transfer, require_completion=True)
    assert result[0] is (state != 'ERR')
    assert len(rows) == 1
    assert rows[0]['status'] == ('failed' if state == 'ERR' else 'success')
    if state == 'ERR':
        assert rows[0]['bytes'] is None
        assert 'native_state:ERR' in rows[0]['reasons']


def test_generic_capture_absent_when_disabled():
    _, progress, rows = setup_capture()
    progress._capture_writer = None
    transfer = submit(progress)
    assert progress.poll_transfer(transfer, require_completion=True)[0]
    assert rows == []


def test_generic_capture_does_not_duplicate_remote_context(monkeypatch):
    _, progress, rows = setup_capture()
    context = SimpleNamespace(timings={}, native_state=None)
    monkeypatch.setattr(operation_capture, 'current', lambda: context)
    # Bypass the decorator, which belongs to the existing remote context observer.
    transfer, _ = progress.submit_transfer.__wrapped__(
        progress, 'WRITE', (_mem(0),), (_mem(0, owner='remote-agent'),),
        remote_side_agent='remote-agent')
    assert progress.poll_transfer(transfer, require_completion=True)[0]
    assert rows == []
    assert 'native_handle_released_ns' in context.timings
