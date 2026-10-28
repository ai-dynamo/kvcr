"""Bounded metadata observations. READY snapshots are never read leases.

Caller and progress share only this lock, never the caller's state lock. All
wire I/O and source index inspection run on progress. No native operation IDs,
reader claims, framework callbacks or policy accounting are used here.
"""
import math
import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .types import CacheTier, QueryStatus

BATCH = 256
FRAME_KEY_BYTES = 65536
CONCURRENT = 16
BUDGET = 16
MAX_REQUESTS = 128
MAX_REQUEST_KEYS = 524288
MAX_KEYS = 72 * MAX_REQUEST_KEYS
# Even single-key calls can queue without treating temporary wire capacity as
# a model-request error. Logical keys bound memory; only 16 frames can fly.
MAX_QUEUED = MAX_KEYS
MAX_INCOMING = 256
LIFETIME = 5.0
OBSERVATION_TTL = 5.0
RETRY = .01


@dataclass(slots=True)
class _Observation:
    deadline: float
    ready: bool | None = None
    metadata: dict | None = None
    expires: float = 0


@dataclass(slots=True)
class _Request:
    endpoint: str
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    observations: dict = field(default_factory=dict)


@dataclass(slots=True)
class _Probe:
    request_id: str
    generation: str
    endpoint: str
    keys: tuple
    deadline: float
    probe_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    incarnation: str | None = None
    epoch: int = 0
    retry_at: float = 0


class ReadyQueries:
    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.RLock()
        self.requests = OrderedDict()
        self.queued = deque()
        self.active = OrderedDict()
        self.incoming = deque()
        self.peers = {}
        self.key_count = 0
        self.closed = False

    def discard(self, request_id):
        with self.lock:
            request = self.requests.pop(request_id, None)
            if request is not None:
                self.key_count -= len(request.observations)
                if not any(r.endpoint == request.endpoint for r in self.requests.values()):
                    self.peers.pop(request.endpoint, None)
            self.queued = deque(p for p in self.queued if p.request_id != request_id)
            self.active = OrderedDict((i, p) for i, p in self.active.items()
                                      if p.request_id != request_id)

    def close(self):
        with self.lock:
            self.closed = True
            self.requests.clear()
            self.queued.clear()
            self.active.clear()
            self.incoming.clear()
            self.peers.clear()
            self.key_count = 0

    def _now(self):
        return self.backend._kvcr._clock()

    def _current(self, request, observation):
        metadata = observation.metadata
        if metadata is None or metadata.get('source_incarnation') is None:
            return True
        return self._peer(request.endpoint) == (metadata['source_incarnation'], metadata['source_progress_epoch'])

    def _peer(self, endpoint):
        d = self.backend._dangling_ops
        native = d.sources.get(endpoint)
        cached = self.peers.get(endpoint)
        if cached is None:
            return native, d.source_progress_epochs.get((endpoint, native), 0)
        incarnation, epoch, native_at_observation = cached
        if native is not None and native != native_at_observation and native != incarnation:
            return native, d.source_progress_epochs.get((endpoint, native), 0)
        if native == incarnation:
            epoch = max(epoch, d.source_progress_epochs.get((endpoint, native), 0))
        return incarnation, epoch

    def _remember_peer(self, endpoint, incarnation, epoch):
        # Bound metadata-only peer knowledge by live requests. Never grow native
        # peer maps or modify accepted writes, registrations, routes or fences.
        self.peers[endpoint] = (incarnation, epoch, self.backend._dangling_ops.sources.get(endpoint))

    def query(self, keys, request_id, endpoint, timeout_ms):
        with self.lock:
            if self.closed:
                return [(QueryStatus.MISS, None)] * len(keys)
            now = self._now()
            request = self.requests.get(request_id)
            if request is None:
                # Reclaim only wholly expired, settled request state. Pending
                # work retains its original deadline and is never displaced.
                for old_id, old in tuple(self.requests.items()):
                    if old.observations and all(o.ready is not None and o.expires <= now
                                               for o in old.observations.values()):
                        self.discard(old_id)
                if len(self.requests) >= MAX_REQUESTS:
                    raise BufferError('READY query request bound exceeded')
                request = self.requests[request_id] = _Request(endpoint)
            if request.endpoint != endpoint:
                raise ValueError('READY query endpoint changed without hint invalidation')
            for key in keys:
                o = request.observations.get(key)
                if o is not None and o.ready is not None and not self._current(request, o):
                    o.expires = now
            new = tuple(dict.fromkeys(k for k in keys if k not in request.observations
                or (request.observations[k].ready is not None
                    and request.observations[k].expires <= now)))
            added = sum(k not in request.observations for k in new)
            if (len(request.observations) + added > MAX_REQUEST_KEYS
                    or self.key_count + added > MAX_KEYS):
                raise BufferError('READY query key/queue bound exceeded')
            deadline = now + min(timeout_ms / 1000, LIFETIME)
            batches, batch, size = [], [], 0
            for key in new:
                if len(batch) == BATCH or size + len(key) > FRAME_KEY_BYTES:
                    batches.append(tuple(batch))
                    batch, size = [], 0
                batch.append(key)
                size += len(key)
            if batch:
                batches.append(tuple(batch))
            if len(self.queued) + len(batches) > MAX_QUEUED:
                raise BufferError('READY query key/queue bound exceeded')
            for batch in batches:
                self.queued.append(_Probe(request_id, request.generation, endpoint, batch, deadline))
                for key in batch:
                    request.observations[key] = _Observation(deadline)
            self.key_count += added
            result = []
            for key in keys:
                observation = request.observations[key]
                if observation.ready is None and now >= observation.deadline:
                    self._finish(request, key, False, now, reason='deadline')
                if observation.ready is None:
                    result.append((QueryStatus.FETCHING, CacheTier.REMOTE_G2))
                else:
                    result.append((QueryStatus.FETCHABLE, CacheTier.REMOTE_G2)
                                  if observation.ready else (QueryStatus.MISS, None))
            return result

    def observations(self, keys, request_id):
        with self.lock:
            request = self.requests.get(request_id)
            now = self._now()
            return [dict(o.metadata) if request and (o := request.observations.get(key))
                    and o.metadata is not None and o.expires > now and self._current(request, o)
                    else None for key in keys]

    def _finish(self, request, key, ready, now, *, expected_deadline=None, **metadata):
        o = request.observations.get(key)
        if (o is None or o.ready is not None
                or (expected_deadline is not None and o.deadline != expected_deadline)):
            return
        o.ready = ready
        # The pending deadline and observation lifetime are separate. The latter
        # starts only on receipt and is NEVER extended by cache reads/requeries.
        o.expires = now + OBSERVATION_TTL
        o.metadata = dict(request_generation=request.generation, source_endpoint=request.endpoint,
            observed_at_monotonic=now, valid_until_monotonic=o.expires,
            status='FETCHABLE' if ready else 'MISS', **metadata)

    def receive(self, payload):
        keys = payload.get('keys')
        if (not isinstance(keys, list) or not 0 < len(keys) <= BATCH
                or any(not isinstance(k, bytes) or not 0 < len(k) <= 4096 for k in keys)
                or sum(map(len, keys)) > FRAME_KEY_BYTES):
            return
        for name in ('probe_id', 'request_generation'):
            if not isinstance(payload.get(name), str) or not 0 < len(payload[name]) <= 128:
                return
        endpoint = payload.get('sender_control_endpoint')
        if not isinstance(endpoint, str) or not 0 < len(endpoint) <= 1024:
            return
        for name in ('source_progress_epoch', 'requested_epoch', 'sender_progress_epoch', 'export_sequence'):
            value = payload.get(name)
            if value is not None and (type(value) is not int or not 0 <= value < 2**64):
                return
        for name in ('deadline', 'remaining_timeout_ms'):
            value = payload.get(name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                return
        state = payload.get('state')
        if state is not None and (not isinstance(state, str) or len(state) > 32):
            return
        for name in ('source_incarnation', 'sender_incarnation', 'requested_incarnation'):
            value = payload.get(name)
            if value is not None and (not isinstance(value, str) or not 0 < len(value) <= 128):
                return
        for name in ('ready', 'generations'):
            value = payload.get(name)
            if value is not None and (not isinstance(value, list) or len(value) > BATCH):
                return
        if payload['type'] == 'ready_probe':
            pools = payload.get('pool_layouts')
            if (not isinstance(pools, list) or not 0 < len(pools) <= 256
                    or any(not isinstance(row, list) or len(row) != 2
                        or not isinstance(row[0], str) or len(row[0]) > 256
                        or type(row[1]) is not int or row[1] <= 0 for row in pools)):
                return
        # Unknown wire fields (including SDK metadata) never occupy the inbox.
        fields = ('type', 'keys', 'probe_id', 'request_generation', 'deadline',
            'sender_control_endpoint', 'source_incarnation', 'sender_incarnation',
            'requested_incarnation', 'source_progress_epoch', 'requested_epoch',
            'sender_progress_epoch', 'remaining_timeout_ms', 'pool_layouts',
            'state', 'ready', 'generations', 'export_sequence')
        payload = {name: payload[name] for name in fields if name in payload
                   and (name != 'pool_layouts' or payload['type'] == 'ready_probe')}
        with self.lock:
            if not self.closed and len(self.incoming) < MAX_INCOMING:
                self.incoming.append(payload)

    def poll(self, progress):
        with self.lock:
            if self.closed:
                return False
            work = False
            for _ in range(min(BUDGET, len(self.incoming))):
                payload = self.incoming.popleft()
                if payload['type'] == 'ready_probe':
                    self._answer(progress, payload)
                else:
                    self._reply(payload)
                work = True
            for _ in range(min(BUDGET, len(self.queued))):
                if len(self.active) >= CONCURRENT:
                    break
                p = self.queued.popleft()
                self.active[p.probe_id] = p
            now = self._now()
            for probe_id, p in tuple(self.active.items()):
                request = self.requests.get(p.request_id)
                if request is None or request.generation != p.generation:
                    self.active.pop(probe_id)
                    continue
                if now >= p.deadline or progress._stop_requested:
                    for key in p.keys:
                        self._finish(request, key, False, now, expected_deadline=p.deadline, reason='deadline')
                    self.active.pop(probe_id)
                    work = True
                    continue
                if now < p.retry_at:
                    continue
                p.retry_at = now + RETRY
                b = self.backend
                incarnation, epoch = self._peer(p.endpoint)
                if p.incarnation is not None and incarnation != p.incarnation:
                    for key in p.keys:
                        self._finish(request, key, False, now, expected_deadline=p.deadline, reason='source_changed')
                    self.active.pop(probe_id)
                    continue
                p.incarnation = incarnation
                p.epoch = epoch
                b._send_control(progress, p.endpoint, dict(type='ready_probe', probe_id=p.probe_id,
                    request_generation=p.generation, keys=p.keys, deadline=p.deadline,
                    remaining_timeout_ms=(p.deadline-now)*1000,
                    source_incarnation=p.incarnation, source_progress_epoch=p.epoch,
                    pool_layouts=b._kvcr.pool_layouts))
                work = True
            return work

    def _answer(self, progress, payload):
        b = self.backend
        keys = payload.get('keys')
        pools = payload.get('pool_layouts')
        endpoint = payload.get('sender_control_endpoint')
        remaining = payload.get('remaining_timeout_ms')
        deadline = payload.get('deadline')
        if (not isinstance(endpoint, str) or not endpoint
                or not isinstance(payload.get('probe_id'), str)
                or not isinstance(payload.get('request_generation'), str)
                or not isinstance(keys, list) or not 0 < len(keys) <= BATCH
                or any(not isinstance(k, bytes) or not 0 < len(k) <= 4096 for k in keys)
                or type(remaining) not in (int, float) or not math.isfinite(remaining)
                or not 0 < remaining <= LIFETIME*1000
                or type(deadline) not in (int, float) or not math.isfinite(deadline)):
            return
        response = dict(type='ready_probe_reply', probe_id=payload['probe_id'],
            request_generation=payload['request_generation'], keys=keys, deadline=deadline,
            requested_epoch=payload.get('source_progress_epoch'),
            requested_incarnation=payload.get('source_incarnation'),
            sender_progress_epoch=b._dangling_ops.source_progress_epoch,
            state='unavailable', ready=[False]*len(keys), generations=[None]*len(keys),
            export_sequence=None)
        if (payload.get('source_incarnation') != b._dangling_ops.incarnation
                or type(payload.get('source_progress_epoch')) is not int
                or payload['source_progress_epoch'] != b._dangling_ops.source_progress_epoch):
            response['state'] = 'stale'
        elif (not b._dangling_ops._source_stalled and b._options.guard_only_sources
                and b._kvcr._local_dram is not None
                and pools == [list(pair) for pair in b._kvcr.pool_layouts]):
            snapshot = b._kvcr._local_dram.probe_export_subset(tuple(keys), tuple(name for name, _ in pools))
            if snapshot is None:
                response['state'] = 'busy'
            else:
                response['state'] = 'ready'
                response['ready'], response['generations'], response['export_sequence'] = snapshot
        b._send_control(progress, endpoint, response)

    def _reply(self, payload):
        probe_id = payload.get('probe_id')
        if not isinstance(probe_id, str):
            return
        p = self.active.get(probe_id)
        if p is None:
            return
        request = self.requests.get(p.request_id)
        now = self._now()
        if (request is None or request.generation != p.generation or now >= p.deadline
                or payload.get('sender_control_endpoint') != p.endpoint
                or payload.get('request_generation') != p.generation
                or payload.get('keys') != list(p.keys)
                or payload.get('deadline') != p.deadline
                or payload.get('requested_incarnation') != p.incarnation
                or type(payload.get('requested_epoch')) is not int
                or payload.get('requested_epoch') != p.epoch):
            return
        incarnation = payload.get('sender_incarnation')
        epoch = payload.get('sender_progress_epoch')
        if (not isinstance(incarnation, str) or not incarnation
                or type(epoch) is not int or epoch < 0):
            return
        if incarnation != p.incarnation:
            if payload.get('state') != 'stale':
                return
            # Challenge only: never authorize from a discovery response. This
            # does not import NIXL metadata, replace routes or touch native refs.
            current, _ = self._peer(p.endpoint)
            if current not in (None, p.incarnation, incarnation):
                return
            self._remember_peer(p.endpoint, incarnation, epoch)
            p.incarnation = incarnation
            return
        if epoch < p.epoch:
            return
        if payload.get('state') == 'stale':
            self._remember_peer(p.endpoint, incarnation, epoch)
            return
        if epoch != p.epoch:
            return
        if payload.get('state') == 'busy':
            return
        if payload.get('state') not in ('ready', 'unavailable'):
            return
        flags, generations, sequence = (payload.get(name) for name in ('ready', 'generations', 'export_sequence'))
        if (not isinstance(flags, list) or len(flags) != len(p.keys)
                or any(type(v) is not bool for v in flags)
                or not isinstance(generations, list) or len(generations) != len(p.keys)
                or any(g is not None and (type(g) is not int or g <= 0) for g in generations)
                or any(flag and g is None for flag, g in zip(flags, generations))
                or (payload['state'] == 'ready' and (type(sequence) is not int or sequence < 0))
                or (payload['state'] == 'unavailable' and any(flags))):
            return
        for key, flag, generation in zip(p.keys, flags, generations):
            self._finish(request, key, flag, now, expected_deadline=p.deadline, source_incarnation=p.incarnation,
                source_progress_epoch=epoch, export_sequence=sequence, residency_generation=generation,
                reason='ready' if flag else ('source_absent_or_incompatible'
                    if payload['state']=='ready' else 'source_unavailable'))
        self.active.pop(probe_id)
