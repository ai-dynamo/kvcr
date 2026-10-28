"""Explicit compact lifecycle capture; producer threads never perform file I/O.

JSON schema 2 plus KVCRKEY1/length/CRC32/MessagePack exact-key journal.
Per-thread buffers have a fixed byte budget; overflow makes evidence invalid.
"""
import atexit
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
import uuid
import zlib

import msgspec

SCHEMA_VERSION = 2
MAGIC = b'KVCRKEY1'
FRAME = struct.Struct('>II')
BUFFER_BYTES = 32 * 1024 * 1024


class CaptureInvalid(RuntimeError):
    pass


class _Buffer:
    def __init__(self, size):
        self.data = bytearray(size)
        self.used = 0
        self.lock = threading.Lock()
        self.queries = Counter()
        self.turns = {}
        self.interval_start_ns = time.time_ns()


class BufferedWriter:
    def __init__(self, root, *, source_archive_sha256=None, buffer_bytes=BUFFER_BYTES,
                 start_thread=True, health_dir=None, manifest_path=None, max_threads=32):
        self.root = Path(root)
        self.pid, self.session = os.getpid(), uuid.uuid4().hex
        self.path = self.root / f'source-{self.pid}-{self.session}.jsonl'
        self.key_path = self.root / f'keys-{self.pid}-{self.session}.bin'
        self.health_dir = Path(health_dir) if health_dir else None
        self.source_sha = source_archive_sha256 or os.environ.get('KVCR_SOURCE_ARCHIVE_SHA256')
        self.manifest_path = manifest_path or os.environ.get('KVCR_SOURCE_MANIFEST')
        self.buffer_bytes, self.max_threads = buffer_bytes, max_threads
        self.buffers, self.identities = {}, set()
        self.lock = threading.Lock()
        self.dropped = 0
        self.drop_reasons = Counter()
        self.error = None
        self.instrumentation_errors = []
        self.closed = False
        self.accepting = True
        self.sequence = self.key_sequence = 0
        self.query_attempts = self.active_turns = self.idle_turns = 0
        self.stop = threading.Event()
        self.thread = None
        if start_thread:
            self.thread = threading.Thread(target=self._run, name='kvcr-capture', daemon=True)
            self.thread.start()

    def _buffer(self):
        thread = threading.get_ident()
        with self.lock:
            if not self.accepting:
                self.dropped += 1
                self.drop_reasons['admission_closed'] += 1
                return None
            buffer = self.buffers.get(thread)
            if buffer is None:
                if len(self.buffers) >= self.max_threads:
                    self.dropped += 1
                    self.drop_reasons['max_threads'] += 1
                    return None
                buffer = self.buffers[thread] = _Buffer(self.buffer_bytes)
            return buffer

    def _enqueue(self, channel, record):
        if self.closed or self.error:
            with self.lock:
                self.dropped += 1
                self.drop_reasons['closed_or_error'] += 1
            return False
        buffer = self._buffer()
        if buffer is None:
            return False
        try:
            body = msgspec.msgpack.encode((channel, record))
            size = len(body) + FRAME.size
            with buffer.lock:
                if not self.accepting or size > len(buffer.data) - buffer.used:
                    with self.lock:
                        self.dropped += 1
                        self.drop_reasons['admission_closed' if not self.accepting else 'buffer_full'] += 1
                    return False
                FRAME.pack_into(buffer.data, buffer.used, len(body), zlib.crc32(body))
                offset = buffer.used + FRAME.size
                buffer.data[offset:offset + len(body)] = body
                buffer.used += size
            return True
        except Exception as error:
            self.invalidate(f'enqueue: {type(error).__name__}')
            return False

    def emit(self, record):
        return self._enqueue(0, {'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), **record})

    def journal(self, record):
        return self._enqueue(1, {'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), **record})

    def invalidate(self, reason):
        with self.lock:
            if len(self.instrumentation_errors) < 16:
                self.instrumentation_errors.append(reason)

    def identity_ready(self, agent):
        with self.lock:
            if agent in self.identities:
                return
            self.identities.add(agent)
        self.emit(dict(event='identity_ready', agent=agent, identity=agent,
                       role='guard' if str(agent).startswith('KVCR-Guard-') else 'worker'))

    def query_counts(self, agent, statuses):
        buffer = self._buffer()
        if buffer:
            with buffer.lock:
                if not self.accepting:
                    with self.lock:
                        self.dropped += 1
                        self.drop_reasons['admission_closed'] += 1
                    return
                for status, tier in statuses:
                    buffer.queries[(agent, status.name, tier.name if tier is not None else 'NONE')] += 1

    def progress_turn(self, agent, active, wall_ns, cpu_ns, start_gap_ns, *,
                      end_to_start_gap_ns=None, observed_work=None, inflight=False):
        buffer = self._buffer()
        if buffer:
            with buffer.lock:
                if not self.accepting:
                    with self.lock:
                        self.dropped += 1
                        self.drop_reasons['admission_closed'] += 1
                    return
                item = buffer.turns.get(agent)
                if item is None:
                    item = buffer.turns[agent] = dict(active_turns=0, idle_turns=0,
                        work_turns=0, no_work_turns=0, inflight_turns=0,
                        active_wall_ns=0, idle_wall_ns=0, active_cpu_ns=0, idle_cpu_ns=0,
                        max_start_gap_ns=0, max_end_to_start_gap_ns=0,
                        **{kind + field: [0] * 7 for kind in ('', 'active_', 'idle_')
                           for field in ('wall_histogram', 'cpu_histogram', 'start_gap_histogram', 'end_to_start_gap_histogram')})
                kind = 'active' if active else 'idle'
                work = active if observed_work is None else observed_work
                item['work_turns' if work else 'no_work_turns'] += 1
                item['inflight_turns'] += bool(inflight)
                item[kind + '_turns'] += 1
                item[kind + '_wall_ns'] += wall_ns
                item[kind + '_cpu_ns'] += cpu_ns
                if start_gap_ns is not None:
                    item['max_start_gap_ns'] = max(item['max_start_gap_ns'], start_gap_ns)
                if end_to_start_gap_ns is not None:
                    item['max_end_to_start_gap_ns'] = max(item['max_end_to_start_gap_ns'], end_to_start_gap_ns)
                for field, value in [('wall_histogram', wall_ns), ('cpu_histogram', cpu_ns),
                                     ('start_gap_histogram', start_gap_ns),
                                     ('end_to_start_gap_histogram', end_to_start_gap_ns)]:
                    if value is None:
                        continue
                    index = sum(value > bound for bound in (10000, 100000, 1000000, 10000000, 100000000, 1000000000))
                    item[field][index] += 1
                    item[kind + '_' + field][index] += 1

    def _health(self):
        if self.health_dir:
            with self.lock:
                buffers, identities = list(self.buffers.values()), sorted(self.identities)
            self.health_dir.mkdir(parents=True, exist_ok=True)
            path = self.health_dir / f'kvcr-{self.pid}-{self.session}.json'
            temporary = path.with_suffix('.partial')
            temporary.write_text(json.dumps(dict(schema_version=2, pid=self.pid, session=self.session,
                capture_path=str(self.path), valid=not self.error and not self.dropped and not self.instrumentation_errors,
                error=self.error, dropped=self.dropped, unflushed=sum(b.used for b in buffers),
                written=self.sequence, closed=self.closed, wall_ns=time.time_ns(), identities=identities)) + '\n')
            temporary.replace(path)

    def _run(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            proof = {}
            if self.manifest_path:
                raw = Path(self.manifest_path).read_bytes()
                proof = json.loads(raw)
                self.source_sha = self.source_sha or proof.get('source_archive_sha256')
                if proof.get('source_archive_sha256', self.source_sha) != self.source_sha:
                    self.invalidate('source provenance mismatch')
            if not self.source_sha:
                self.invalidate('source provenance missing')
            with self.path.open('x', encoding='utf-8') as json_file, self.key_path.open('xb') as binary_file:
                binary_file.write(MAGIC)
                def output(channel, record):
                    if channel:
                        self.key_sequence += 1
                        record = dict({'schema_version': 2, 'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), 'pid': self.pid, 'session': self.session,
                                      'sequence': self.key_sequence}, **record)
                        body = msgspec.msgpack.encode(record)
                        binary_file.write(FRAME.pack(len(body), zlib.crc32(body)))
                        binary_file.write(body)
                    else:
                        self.sequence += 1
                        record = dict({'schema_version': 2, 'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), 'pid': self.pid, 'session': self.session,
                                      'sequence': self.sequence}, **record)
                        json_file.write(json.dumps(record, separators=(',', ':')) + '\n')
                header = dict(event='capture_start', source_archive_sha256=self.source_sha,
                    provenance=proof, compact_capture_version=1, key_journal=self.key_path.name)
                output(0, header); output(1, header)
                while True:
                    stopping = self.stop.is_set()
                    with self.lock:
                        if stopping:
                            # Seal admission before the final scan. Every commit
                            # rechecks admission under its buffer lock, so a
                            # producer already encoding cannot append behind us.
                            self.accepting = False
                        buffers = list(self.buffers.values())
                    for buffer in buffers:
                        with buffer.lock:
                            batch = bytes(memoryview(buffer.data)[:buffer.used])
                            buffer.used = 0
                            queries, turns = buffer.queries, buffer.turns
                            buffer.queries, buffer.turns = Counter(), {}
                            interval_start = buffer.interval_start_ns
                            interval_end = buffer.interval_start_ns = time.time_ns()
                        offset = 0
                        while offset < len(batch):
                            length, crc = FRAME.unpack_from(batch, offset)
                            offset += FRAME.size
                            body = batch[offset:offset + length]; offset += length
                            if zlib.crc32(body) != crc:
                                raise CaptureInvalid('in-memory capture CRC mismatch')
                            channel, record = msgspec.msgpack.decode(body)
                            output(channel, record)
                        agents = {key[0] for key in queries}
                        for agent in agents:
                            counts = {f'{status} / {tier}': count for (owner, status, tier), count in queries.items() if owner == agent}
                            self.query_attempts += sum(counts.values())
                            output(0, dict(event='query_outcomes', identity=agent, agent=agent,
                                wall_ns=time.time_ns(), query_status_tier_counts=counts, denominator=sum(counts.values()),
                                interval_start_ns=interval_start, interval_end_ns=interval_end,
                                scope='Aggregated observations since preceding flush; no per-query JSON'))
                        for agent, turn in turns.items():
                            self.active_turns += turn['active_turns']
                            self.idle_turns += turn['idle_turns']
                            output(0, dict(event='progress_schedule', agent=agent, identity=agent,
                                wall_ns=time.time_ns(), interval_start_ns=interval_start, interval_end_ns=interval_end, **turn))
                    json_file.flush(); binary_file.flush()
                    self._health()
                    if stopping:
                        break
                    self.stop.wait(.02)
                # Stop observed after a previous scan requires the additional
                # sealed scan above, including newly registered thread buffers.
                unflushed = 0
                for buffer in buffers:
                    with buffer.lock:
                        unflushed += buffer.used
                        if buffer.queries or buffer.turns:
                            self.invalidate('undrained aggregate counters at shutdown')
                valid = not self.error and not self.dropped and not self.instrumentation_errors and unflushed == 0
                output(1, dict(event='capture_end', valid=valid, dropped=self.dropped,
                    drop_reasons=dict(self.drop_reasons),
                    unflushed=unflushed,
                    error=self.error, instrumentation_errors=list(self.instrumentation_errors),
                    written_before_footer=self.key_sequence))
                output(0, dict(event='capture_end', valid=valid, dropped=self.dropped, unflushed=unflushed,
                    drop_reasons=dict(self.drop_reasons),
                    error=self.error, instrumentation_errors=list(self.instrumentation_errors),
                    written_before_footer=self.sequence, key_journal_frames=self.key_sequence,
                    aggregate_query_attempts=self.query_attempts, aggregate_active_turns=self.active_turns,
                    aggregate_idle_turns=self.idle_turns))
                self.closed = True
            self._health()
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            try:
                self._health()
            except Exception:
                pass

    def close(self):
        if not self.thread:
            self.thread = threading.Thread(target=self._run, name='kvcr-capture', daemon=True)
            self.thread.start()
        self.stop.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            self.error = 'capture flusher shutdown timeout'
            raise CaptureInvalid(self.error)
        if self.error:
            raise CaptureInvalid(self.error)


def iter_key_journal(path):
    try:
        with Path(path).open('rb') as source:
            if source.read(len(MAGIC)) != MAGIC:
                raise CaptureInvalid('unknown key journal header')
            while header := source.read(FRAME.size):
                if len(header) != FRAME.size:
                    raise CaptureInvalid('truncated key frame header')
                length, crc = FRAME.unpack(header)
                if not 0 < length <= BUFFER_BYTES:
                    raise CaptureInvalid('invalid key frame length')
                body = source.read(length)
                if len(body) != length or zlib.crc32(body) != crc:
                    raise CaptureInvalid('truncated/corrupt key journal')
                yield msgspec.msgpack.decode(body)
    except (OSError, ValueError, msgspec.DecodeError) as error:
        raise CaptureInvalid(f'key journal {path}: {error}') from error


def validate_key_journal(path):
    first = last = None
    counts = Counter()
    number = 0
    for number, row in enumerate(iter_key_journal(path), 1):
        if (not isinstance(row, dict) or not {'schema_version','pid','session','sequence','event'}.issubset(row)
                or row.get('schema_version') != 2 or row.get('sequence') != number):
            raise CaptureInvalid('invalid key journal sequence/schema')
        if first is None:
            first = row
            if row.get('event') != 'capture_start':
                raise CaptureInvalid('missing key header')
        if any(row.get(k) != first.get(k) for k in ('pid', 'session')):
            raise CaptureInvalid('mixed key journal PID/session')
        counts[row['event']] += 1
        last = row
    if (not last or last.get('event') != 'capture_end' or last.get('valid') is not True
            or last.get('dropped') or last.get('unflushed') or last.get('error') or last.get('instrumentation_errors')
            or last.get('written_before_footer') != number - 1):
        raise CaptureInvalid('missing/invalid key journal footer')
    return dict(valid=True, frames=number, counts=dict(counts), pid=first['pid'], session=first['session'])


def validate_capture(root, expected_identities=(), workers=1):
    """Streaming JSON and binary checks; return details/counts, never all rows."""
    paths = sorted(Path(root).glob('source-*.jsonl'))
    if not paths:
        raise CaptureInvalid('no per-PID captures')
    ready, identities, counts, files = set(), set(), Counter(), []
    for path in paths:
        first = last = None
        operations = {}
        number = 0
        try:
            with path.open() as stream:
                for number, line in enumerate(stream, 1):
                    if not line.endswith('\n'):
                        raise CaptureInvalid('missing final newline')
                    row = json.loads(line)
                    if (not isinstance(row, dict) or row.get('schema_version') != 2
                            or row.get('sequence') != number):
                        raise CaptureInvalid('unknown schema/sequence gap')
                    if first is None:
                        first = row
                        if row.get('event') != 'capture_start' or not row.get('source_archive_sha256'):
                            raise CaptureInvalid('missing capture header/source provenance')
                    if any(row.get(k) != first.get(k) for k in ('pid', 'session')):
                        raise CaptureInvalid('mixed PID/session')
                    event = row['event']
                    if event == 'capture_error' or row.get('reason') == 'legacy_completion_callback':
                        raise CaptureInvalid('instrumentation/legacy capture failure')
                    if row.get('identity'):
                        identities.add(row['identity'])
                    if event == 'identity_ready' and row.get('agent'):
                        ready.add(row['identity'])
                    if event in ('operation_start', 'source_nixl_terminal', 'target_terminal'):
                        key = row['operation_key']
                        bit = 1 if event == 'operation_start' else 2
                        if operations.get(key, 0) & bit:
                            raise CaptureInvalid('duplicate operation lifecycle')
                        operations[key] = operations.get(key, 0) | bit
                    if event == 'source_nixl_terminal' and (row.get('guard_only_sources') is not True
                            or row.get('initial_caller_handoff') != 'absent_by_guard_only_progress_path'
                            or row.get('has_framework_pins') is not False):
                        raise CaptureInvalid('missing strict Guard-only source proof')
                    counts[event] += 1
                    last = row
            if (not last or last.get('event') != 'capture_end' or last.get('valid') is not True
                    or last.get('dropped') or last.get('unflushed') or last.get('error')
                    or last.get('instrumentation_errors') or last.get('written_before_footer') != number - 1
                    or any(value != 3 for value in operations.values())):
                raise CaptureInvalid('invalid/unflushed footer or unmatched lifecycle')
            key_name = first.get('key_journal', '')
            if Path(key_name).name != key_name or not key_name:
                raise CaptureInvalid('invalid key journal reference')
            journal = validate_key_journal(path.parent / key_name)
            if (journal['pid'] != first['pid'] or journal['session'] != first['session']
                    or journal['frames'] != last.get('key_journal_frames')):
                raise CaptureInvalid('key journal/capture footer mismatch')
            files.append(dict(path=str(path), pid=first['pid'], session=first['session'], events=number,
                              key_journal=key_name, key_frames=journal['frames']))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CaptureInvalid(f'{path}:{number}: {error}') from error
    missing = set(expected_identities) - (ready & identities)
    if missing:
        raise CaptureInvalid(f'missing identity ready proof: {sorted(missing)}')
    return dict(schema_version=2, valid=True, events=sum(counts.values()), counts=dict(counts),
                identities=sorted(identities), ready_identities=sorted(ready), files=files,
                validator_version='compact-v1')


_writer = None
_writer_lock = threading.Lock()
_mp_finalizer = None


def writer():
    global _writer, _mp_finalizer
    if _writer is not None and _writer.pid == os.getpid():
        return _writer
    root = os.environ.get('KVCR_TELEMETRY_DIR')
    if not root:
        run = os.environ.get('RUN_DIR')
        root = str(Path(run) / 'nixl-transfers') if run else None
    if not root:
        return None
    with _writer_lock:
        if _writer is None or _writer.pid != os.getpid():
            _writer = BufferedWriter(root, health_dir=os.environ.get('KVCR_TELEMETRY_HEALTH_DIR'))
            from multiprocessing.util import Finalize
            _mp_finalizer = Finalize(None, _shutdown, exitpriority=-1000)
        return _writer


def emit(record):
    capture = writer()
    return capture.emit(record) if capture else False


def _shutdown():
    if _writer and _writer.pid == os.getpid():
        _writer.close()


def _error_message(error):
    return str(error)


def _after_fork():
    global _writer, _writer_lock, _mp_finalizer
    _writer = None
    _mp_finalizer = None
    _writer_lock = threading.Lock()


os.register_at_fork(after_in_child=_after_fork)
def _python_shutdown():
    if _mp_finalizer is None or not _mp_finalizer.still_active():
        _shutdown()


atexit.register(_python_shutdown)
