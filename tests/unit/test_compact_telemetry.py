"""CPU-only compact capture and binary lineage regressions."""
import json
from pathlib import Path
import threading

import pytest


def _module():
    try:
        from kvcr import compact_telemetry
    except ImportError:
        pytest.fail('compact telemetry module missing')
    return compact_telemetry


def test_compact_shared_pid_identities_and_exact_keys(tmp_path):
    m = _module()
    output = m.BufferedWriter(tmp_path, source_archive_sha256='test-source', start_thread=False)
    keys = tuple(bytes([i % 256]) * 33 + i.to_bytes(4, 'big') for i in range(1000))
    for agent in ('worker', 'guard-0', 'guard-1'):
        output.identity_ready(agent)
    output.emit(dict(event='operation_start', operation_key='op', identity='worker', agent='worker'))
    output.journal(dict(event='source_get', agent='worker', keys=keys, ready_indices=[0, 999],
                        generations=[1, 2], reasons=['ready'] * len(keys)))
    output.emit(dict(event='source_nixl_terminal', operation_key='op', identity='worker', agent='worker',
                    guard_only_sources=True, initial_caller_handoff='absent_by_guard_only_progress_path',
                    has_framework_pins=False, result='success', bytes=16, blocks=1))
    output.close()
    result = m.validate_capture(tmp_path, ['worker', 'guard-0', 'guard-1'])
    assert result['valid'] and len(result['files']) == 1
    assert result['counts']['source_nixl_terminal'] == 1
    frames = list(m.iter_key_journal(output.key_path))
    assert next(r for r in frames if r['event'] == 'source_get')['keys'] == list(keys)
    assert m.validate_key_journal(output.key_path)['valid']


def test_bounded_overflow_is_explicit_and_invalid(tmp_path):
    m = _module()
    output = m.BufferedWriter(tmp_path, source_archive_sha256='test', buffer_bytes=256, start_thread=False)
    for _ in range(10):
        output.journal(dict(event='source_get', keys=[b'x' * 300]))
    assert output.dropped > 0
    assert output.drop_reasons['buffer_full'] == output.dropped
    assert sum(len(b.data) for b in output.buffers.values()) <= 256
    output.close()
    assert json.loads(output.path.read_text().splitlines()[-1])['drop_reasons']['buffer_full'] == output.dropped
    with pytest.raises(m.CaptureInvalid):
        m.validate_capture(tmp_path)


def test_binary_corruption_and_footer_loss_rejected(tmp_path):
    m = _module()
    output = m.BufferedWriter(tmp_path, source_archive_sha256='test', start_thread=False)
    output.identity_ready('worker')
    output.journal(dict(event='ready', agent='worker', key=b'exact', generation=4, reason='fill_complete'))
    output.close()
    original = output.key_path.read_bytes()
    for bad in (original[:-1], original[:20] + bytes([original[20] ^ 1]) + original[21:]):
        output.key_path.write_bytes(bad)
        with pytest.raises(m.CaptureInvalid):
            m.validate_capture(tmp_path, ['worker'])
    output.key_path.write_bytes(original)


def test_flush_io_occurs_only_on_flusher_thread(tmp_path, monkeypatch):
    m = _module()
    callers = []
    original = Path.open
    def observed(path, *args, **kwargs):
        if args and any(flag in args[0] for flag in ('w', 'a', 'x')):
            callers.append(threading.current_thread().name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', observed)
    output = m.BufferedWriter(tmp_path, source_archive_sha256='test')
    output.identity_ready('worker')
    output.close()
    assert callers and all(name == 'kvcr-capture' for name in callers)
    assert m.validate_capture(tmp_path, ['worker'])['valid']


def test_missing_source_provenance_invalidates_capture(tmp_path):
    m = _module()
    output = m.BufferedWriter(tmp_path, start_thread=False)
    output.identity_ready('worker')
    output.close()
    with pytest.raises(m.CaptureInvalid, match='provenance'):
        m.validate_capture(tmp_path, ['worker'])


def test_aggregate_interval_bounds_and_no_per_query_wallclock(tmp_path, monkeypatch):
    from kvcr.types import CacheTier, QueryStatus
    m = _module()
    clock, calls = [100], []
    def wall():
        calls.append(1)
        return clock[0]
    monkeypatch.setattr(m.time, 'time_ns', wall)
    output = m.BufferedWriter(tmp_path, source_archive_sha256='test', start_thread=False)
    output.identity_ready('worker')
    before = len(calls)
    for _ in range(1000):
        output.query_counts('worker', [(QueryStatus.HIT, CacheTier.LOCAL_G2)])
    assert len(calls) == before
    output.progress_turn('worker', True, 100, 50, 200)
    clock[0] = 200
    output.close()
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    queries = [r for r in rows if r['event'] == 'query_outcomes']
    assert len(queries) == 1 and queries[0]['denominator'] == 1000
    for row in rows:
        assert 'wall_ns' in row
        if row['event'] in ('query_outcomes', 'progress_schedule'):
            assert (row['interval_start_ns'], row['interval_end_ns']) == (100, 200)
    assert rows[-1]['aggregate_query_attempts'] == 1000
    assert rows[-1]['aggregate_active_turns'] == 1


def _child_capture(root):
    from kvcr import compact_telemetry
    compact_telemetry.writer().identity_ready('child')


def test_multiprocessing_exit_drains_capture(tmp_path, monkeypatch):
    import multiprocessing
    m = _module()
    monkeypatch.setenv('KVCR_TELEMETRY_DIR', str(tmp_path))
    monkeypatch.setenv('KVCR_SOURCE_ARCHIVE_SHA256', 'test')
    process = multiprocessing.get_context('fork').Process(target=_child_capture, args=(str(tmp_path),))
    process.start(); process.join(5)
    assert process.exitcode == 0
    assert m.validate_capture(tmp_path, ['child'])['valid']


def test_compact_progress_separates_inflight_work_and_idle_gaps(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kvcr import operation_capture as capture
    m = _module()
    output = m.BufferedWriter(tmp_path, source_archive_sha256='test', start_thread=False)
    progress = SimpleNamespace(_capture_writer=output, _nixl_agent_name='worker', _active_transfers={1: object()}, _in_flight_ops={1: object()})
    clock = [100]
    monkeypatch.setattr(capture.time, 'monotonic_ns', lambda: clock[0])
    @capture.progress_turn
    def iteration(progress):
        clock[0] += 10
        return False
    assert iteration(progress) is False
    clock[0] = 200
    progress._active_transfers.clear(); progress._in_flight_ops.clear()
    assert iteration(progress) is False
    output.close()
    row = next(json.loads(line) for line in output.path.read_text().splitlines() if json.loads(line)['event'] == 'progress_schedule')
    assert row['active_turns'] == 1 and row['idle_turns'] == 1
    assert row['inflight_turns'] == 1 and row['work_turns'] == 0 and row['no_work_turns'] == 2
    assert row['max_start_gap_ns'] == 100
    assert row['max_end_to_start_gap_ns'] == 90
    assert sum(row['active_wall_histogram']) == 1 and sum(row['idle_wall_histogram']) == 1
    assert sum(row['active_cpu_histogram']) == 1 and sum(row['idle_cpu_histogram']) == 1
    assert sum(row['active_start_gap_histogram']) == 0 and sum(row['idle_start_gap_histogram']) == 1
    assert sum(row['active_end_to_start_gap_histogram']) == 0 and sum(row['idle_end_to_start_gap_histogram']) == 1
    assert row['interval_start_ns'] <= row['interval_end_ns']


def test_compact_shutdown_drains_post_scan_quiescent_records(tmp_path):
    from kvcr.types import QueryStatus, CacheTier
    m = _module()
    output = m.BufferedWriter(tmp_path, source_archive_sha256='test', start_thread=False)
    output.identity_ready('worker')
    scanned, resume = threading.Event(), threading.Event()
    health = output._health
    first = [True]
    def paused_health():
        health()
        if first[0]:
            first[0] = False
            scanned.set()
            assert resume.wait(5)
    output._health = paused_health
    output.thread = threading.Thread(target=output._run, name='kvcr-capture', daemon=True)
    output.thread.start()
    errors = []
    def close():
        try:
            output.close()
        except BaseException as error:
            errors.append(error)
    closer = threading.Thread(target=close)
    try:
        assert scanned.wait(5)
        assert output.emit(dict(event='operation_start', operation_key='late', identity='worker', agent='worker'))
        assert output.emit(dict(event='source_nixl_terminal', operation_key='late', identity='worker', agent='worker',
            guard_only_sources=True, initial_caller_handoff='absent_by_guard_only_progress_path', has_framework_pins=False))
        accepted = []
        def last_producer():
            accepted.append(output.journal(dict(event='ready', key=b'late-exact-key', agent='worker', generation=1)))
            output.query_counts('worker', [(QueryStatus.HIT, CacheTier.LOCAL_G2)])
            output.progress_turn('worker', True, 10, 5, 20)
        producer = threading.Thread(target=last_producer)
        producer.start(); producer.join(5)
        assert not producer.is_alive() and accepted == [True]
        # All producers have stopped before close starts. The flusher has not
        # scanned the above JSON records or this newly created thread buffer.
        closer.start()
        assert output.stop.wait(5)
    finally:
        resume.set()
        if closer.ident is None:
            closer.start()
        closer.join(5)
    assert not closer.is_alive() and not errors
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    assert [row['event'] for row in rows].count('operation_start') == 1
    assert [row['event'] for row in rows].count('source_nixl_terminal') == 1
    assert next(row for row in m.iter_key_journal(output.key_path) if row['event'] == 'ready')['key'] == b'late-exact-key'
    assert all(buffer.used == 0 and not buffer.queries and not buffer.turns for buffer in output.buffers.values())
    assert rows[-1]['valid'] and rows[-1]['unflushed'] == 0
    binary_footer = list(m.iter_key_journal(output.key_path))[-1]
    assert binary_footer['valid'] and binary_footer['unflushed'] == 0
    assert rows[-1]['aggregate_query_attempts'] == 1 and rows[-1]['aggregate_active_turns'] == 1
    assert m.validate_capture(tmp_path, ['worker'])['valid']


def test_compact_overlapping_owner_progress_cpu_is_per_thread(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kvcr import operation_capture as capture
    m = _module()
    output = m.BufferedWriter(tmp_path, source_archive_sha256='test', start_thread=False)
    backend = SimpleNamespace(_kvcr=SimpleNamespace(nixl_agent_name='target'),
        _dangling_ops=SimpleNamespace(incarnation='inc'), _options=SimpleNamespace(guard_only_sources=False))
    context = capture.Context(output, backend, 'target', ())
    clock = threading.local()
    monkeypatch.setattr(capture.time, 'thread_time_ns', lambda: clock.value)
    owner_entered, progress_entered = threading.Event(), threading.Event()
    owner_left, finish_called, release_progress = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def owner():
        try:
            clock.value = 1_000_000_000
            context.enter(); context.enter()  # same-thread nesting counts once
            owner_entered.set()
            assert progress_entered.wait(5) and finish_called.wait(5)
            clock.value += 30
            context.leave()
            clock.value += 20
            context.leave()
            owner_left.set()
        except BaseException as error:
            errors.append(error)
            owner_left.set()
    def progress():
        try:
            assert owner_entered.wait(5)
            clock.value = 100
            context.enter()
            progress_entered.set()
            context.finish()
            finish_called.set()
            assert release_progress.wait(5)
            clock.value = 150
            context.leave()
        except BaseException as error:
            errors.append(error)
    threads = [threading.Thread(target=owner), threading.Thread(target=progress)]
    for thread in threads: thread.start()
    try:
        assert owner_left.wait(5)
        assert not context.emitted  # progress interval still open
    finally:
        release_progress.set()
        for thread in threads: thread.join(5)
        output.close()
    assert not errors and all(not thread.is_alive() for thread in threads)
    assert context.cpu_ns == 100 and context.depth == 0
    rows = [json.loads(line) for line in output.path.read_text().splitlines()]
    terminals = [row for row in rows if row['event'] == 'target_terminal']
    assert len(terminals) == 1 and terminals[0]['handling_cpu_ns'] == 100
    assert m.validate_capture(tmp_path)['valid']
