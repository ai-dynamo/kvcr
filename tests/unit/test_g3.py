# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import ctypes
import logging
import os

import pytest
from _kvcr_test_utils import (
    _OPEN_KVCRS,
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    FakeTelemetryStats,
    _decode_control_message,
    _has_outstanding_operations,
    _MatchingHintAdapter,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _write_done_notification,
)

from kvcr import (
    DURATION_METRIC,
    STATE_METRIC,
    TRANSFER_BLOCKS_METRIC,
    TRANSFER_BYTES_METRIC,
)
from kvcr.config import (
    G3Options,
    KVCRConfig,
    LocalDramInfo,
    RemoteFWDramOptions,
)
from kvcr.core import _BlockRecord
from kvcr.local_disk import _G3Residency
from kvcr.local_dram import _LocalDramState
from kvcr.policy import FIFOPolicy, G3FIFOPolicy, G3LRUPolicy
from kvcr.recovery_journal import install_recovery_records
from kvcr.types import (
    BlockKey,
    CacheTier,
    InventoryEvent,
    PlacementAction,
    QueryStatus,
)


class _FakeG3Agent(FakeNixlAgent):
    def __init__(self, *, fail_file_writes: bool = False):
        super().__init__()
        self.state = "DONE"
        self.backends = {}
        self._xfer_backends = {}
        self._file_data = {}
        self._fail_file_writes = fail_file_writes

    def get_plugin_list(self):
        return ["MOCK"]

    def create_backend(self, backend, options):
        self.backends[backend] = dict(options)

    def get_backend_params(self, backend):
        return self.backends[backend]

    def register_memory(self, descs, mem_type="DRAM", backends=None):
        if mem_type == "DRAM" and "MOCK" not in self.backends:
            raise RuntimeError("G3 backend must exist before DRAM registration")
        self.registrations.append((list(descs), mem_type))
        return len(self.registrations)

    def deregister_memory(self, handle, backends=None):
        self.deregistered.append(handle)

    def initialize_xfer(
        self,
        op,
        local_descs,
        remote_descs,
        remote_agent,
        notif_msg=b"",
        backends=None,
    ):
        handle = super().initialize_xfer(
            op,
            local_descs,
            remote_descs,
            remote_agent,
            notif_msg,
            backends,
        )
        self._xfer_backends[handle] = tuple(backends or ())
        return handle

    def transfer(self, handle):
        backend = self._xfer_backends[handle]
        if not backend:
            return super().transfer(handle)
        self.transfers.append(handle)
        operation = self.xfers[handle - 1][0]
        if (
            self._fail_file_writes
            and operation == "WRITE"
            and self._xfer_backends[handle] == ("MOCK",)
        ):
            return "ERR"
        _, local_descs, local_indices, remote_descs, _, _ = self.xfers[handle - 1]
        for index in local_indices:
            local_addr, local_size, _ = local_descs[index]
            file_offset, file_size, file_fd = remote_descs[index]
            size = min(local_size, file_size)
            file_key = (file_fd, file_offset)
            if operation == "WRITE":
                self._file_data[file_key] = ctypes.string_at(local_addr, size)
            else:
                ctypes.memmove(local_addr, self._file_data[file_key], size)
        return self.state


class _MoveLocalToG3Policy(FIFOPolicy):
    def __init__(self):
        self.move = True
        self.failures = []

    def decide_eviction(self, meta, source):
        if self.move and source is CacheTier.LOCAL_G2:
            return (PlacementAction.MOVE_TO, CacheTier.G3)
        return super().decide_eviction(meta, source)

    def decide_recovery(self, meta, failure):
        self.failures.append((meta, failure))
        return super().decide_recovery(meta, failure)


class _RecordingG3LRUPolicy(G3LRUPolicy):
    def __init__(self):
        self.scored = []

    def eviction_score(self, meta, source):
        self.scored.append((meta, source))
        return super().eviction_score(meta, source)


class _RequiresG3Policy(FIFOPolicy):
    required_tiers = frozenset({CacheTier.G3})


def _new_g3_kvcr(
    tmp_path,
    local,
    *,
    agent=None,
    policy=None,
    slot_count=1,
    g3_slot_count=2,
    control=None,
    telemetry=False,
    key_hint_adapter=None,
    remote_options=None,
    inventory_sink=None,
    g3_paths=None,
):
    page_size = os.sysconf("SC_PAGE_SIZE")
    return _new_kvcr(
        agent or _FakeG3Agent(),
        FakePrimaryPinning(),
        control or FakeBytesControl(),
        config=(
            KVCRConfig(nixl_agent_name="target", enable_telemetry=True)
            if telemetry
            else None
        ),
        local_dram=LocalDramInfo(ctypes.addressof(local), len(local), slot_count),
        g3=G3Options(
            paths=((tmp_path / "g3.data",) if g3_paths is None else tuple(g3_paths)),
            capacity_bytes_per_file=page_size * g3_slot_count,
            backend="MOCK",
            backend_options={"threads": "2"},
        ),
        policy=policy,
        inventory_sink=inventory_sink,
        key_hint_adapter=key_hint_adapter,
        remote_options=remote_options,
    )


def _deposit(kvcr, key, address, size):
    handle = kvcr.deposit({key: _mem_descriptor(address, size)})
    return dict(_poll_until(kvcr, bool))[handle][key]


def _metric_totals(stats):
    totals = {}
    for kind, name, value, labels in stats.records:
        key = (kind, name, *labels)
        totals[key] = totals.get(key, 0) + value
    return totals


@pytest.mark.parametrize(
    ("policy", "moved_index"),
    [(G3FIFOPolicy(), 1), (G3LRUPolicy(), 0), (None, 0)],
    ids=["fifo", "lru", "default-lru"],
)
def test_g3_builtin_policy_eviction_order(
    tmp_path,
    policy: FIFOPolicy | None,
    moved_index: int,
) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 3)
    local = ctypes.create_string_buffer(page_size * 2)
    primary_addr = ctypes.addressof(primary)
    events: list[InventoryEvent] = []
    kvcr = _new_g3_kvcr(
        tmp_path,
        local,
        slot_count=2,
        policy=policy,
        inventory_sink=events.append,
    )
    now = 0.0
    kvcr._core._clock = lambda: now
    keys = tuple(BlockKey(f"k{index}".encode()) for index in range(3))

    for index, key in enumerate(keys[:2]):
        assert _deposit(
            kvcr,
            key,
            primary_addr + index * page_size,
            page_size,
        ).success

    now = 1.0
    first_fetch = kvcr.fetch((keys[0],))
    first_claim = dict(kvcr.poll_completed())[first_fetch][keys[0]].release_handle
    now = 2.0
    second_fetch = kvcr.fetch((keys[1],))
    second_claim = dict(kvcr.poll_completed())[second_fetch][keys[1]].release_handle
    assert first_claim is not None and second_claim is not None
    kvcr.release((second_claim, first_claim))

    now = 3.0
    assert _deposit(
        kvcr,
        keys[2],
        primary_addr + 2 * page_size,
        page_size,
    ).success
    statuses = kvcr.query(keys)
    assert statuses.pop(moved_index) == (QueryStatus.FETCHABLE, CacheTier.G3)
    assert statuses == [
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    assert InventoryEvent((keys[moved_index],), CacheTier.G3, False) in events


@pytest.mark.parametrize("policy", [G3FIFOPolicy(), G3LRUPolicy(), _RequiresG3Policy()])
def test_policy_required_tiers_must_be_configured(policy) -> None:
    with pytest.raises(ValueError, match="requires configured G3"):
        _new_kvcr(
            FakeNixlAgent(),
            FakePrimaryPinning(),
            FakeBytesControl(),
            policy=policy,
        )


@pytest.mark.parametrize("alias_kind", ["duplicate", "symlink"])
def test_g3_rejects_paths_that_resolve_to_the_same_location(
    tmp_path, alias_kind
) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    local = ctypes.create_string_buffer(page_size)
    path = tmp_path / "g3.data"
    if alias_kind == "symlink":
        path.touch()
        alias = tmp_path / "g3-link.data"
        alias.symlink_to(path)
    else:
        alias = path

    with pytest.raises(ValueError, match="G3 file paths must be unique"):
        _new_g3_kvcr(tmp_path, local, g3_paths=(path, alias))


def test_g3_rejects_hard_linked_paths_and_releases_the_first_lock(
    tmp_path,
) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    local = ctypes.create_string_buffer(page_size)
    path = tmp_path / "g3.data"
    alias = tmp_path / "g3-hard-link.data"
    path.touch()
    os.link(path, alias)

    with pytest.raises(ValueError, match="must not alias the same file"):
        _new_g3_kvcr(tmp_path, local, g3_paths=(path, alias))

    # A new controller can lock it only if failure cleanup closed the owner FD.
    assert _new_g3_kvcr(tmp_path, local, g3_paths=(path,))._core._g3 is not None


def test_g3_stripes_slots_across_files_and_reuses_an_evicted_slot(
    tmp_path,
) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 5)
    payloads = tuple(bytes((ord("a") + index,)) * page_size for index in range(5))
    primary.raw = b"".join(payloads)
    local = ctypes.create_string_buffer(page_size)
    destination = ctypes.create_string_buffer(page_size * 4)
    agent = _FakeG3Agent()
    paths = (tmp_path / "g3-0.data", tmp_path / "g3-1.data")
    kvcr = _new_g3_kvcr(
        tmp_path,
        local,
        agent=agent,
        g3_paths=paths,
        g3_slot_count=2,
    )
    keys = tuple(BlockKey(f"k{index}".encode()) for index in range(5))
    primary_addr = ctypes.addressof(primary)

    for index, key in enumerate(keys):
        assert _deposit(
            kvcr,
            key,
            primary_addr + index * page_size,
            page_size,
        ).success

    assert kvcr.query(keys) == [
        (QueryStatus.FETCHABLE, CacheTier.G3),
        (QueryStatus.FETCHABLE, CacheTier.G3),
        (QueryStatus.FETCHABLE, CacheTier.G3),
        (QueryStatus.FETCHABLE, CacheTier.G3),
        (QueryStatus.HIT, CacheTier.LOCAL_G2),
    ]
    g3 = kvcr._core._g3
    assert g3 is not None
    first_fd, second_fd = g3._direct_fds
    locations = []
    for key in keys[:4]:
        residency = kvcr._core._block_record_map[key].g3
        assert residency is not None
        descriptor = g3._descriptor(residency.slot)
        locations.append((descriptor.device_Id, descriptor.addr))
    assert locations == [
        (first_fd, 0),
        (second_fd, 0),
        (first_fd, page_size),
        (second_fd, page_size),
    ]

    destination_addr = ctypes.addressof(destination)
    deliver = kvcr.deliver(
        {
            key: _mem_descriptor(destination_addr + index * page_size, page_size)
            for index, key in enumerate(keys[:4])
        }
    )
    deliver_result = dict(_poll_until(kvcr, lambda done: deliver in dict(done)))[
        deliver
    ]
    assert all(entry.success for entry in deliver_result.values())
    assert destination.raw == b"".join(payloads[:4])

    evicted = kvcr._core._block_record_map[keys[1]].g3
    assert evicted is not None
    evicted_slot = evicted.slot
    fetch = kvcr.fetch((keys[0],))
    fetch_result = dict(_poll_until(kvcr, lambda done: fetch in dict(done)))[fetch][
        keys[0]
    ]
    assert fetch_result.success
    assert kvcr.query((keys[1], keys[4])) == [
        (QueryStatus.MISS, None),
        (QueryStatus.FETCHABLE, CacheTier.G3),
    ]
    replacement = kvcr._core._block_record_map[keys[4]].g3
    assert replacement is not None and replacement.slot == evicted_slot
    replacement_destination = ctypes.create_string_buffer(page_size)
    replacement_deliver = kvcr.deliver(
        {keys[4]: _mem_descriptor(ctypes.addressof(replacement_destination), page_size)}
    )
    replacement_result = dict(
        _poll_until(kvcr, lambda done: replacement_deliver in dict(done))
    )[replacement_deliver][keys[4]]
    assert replacement_result.success
    assert replacement_destination.raw == payloads[4]


def test_g3_recovery_rebuilds_free_slots_and_a_tier_recovered_full_frees_one(
    tmp_path,
) -> None:
    """Recovery rebuilds free slots and spills observably evict recovered blocks."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 3)
    local = ctypes.create_string_buffer(page_size)
    destination = ctypes.create_string_buffer(page_size)
    agent = _FakeG3Agent()
    kvcr = _new_g3_kvcr(tmp_path, local, agent=agent, g3_slot_count=3)
    g3 = kvcr._core._g3
    assert g3 is not None
    first, second = BlockKey(b"first"), BlockKey(b"second")

    install_recovery_records(
        kvcr._core,
        {
            first: _BlockRecord(g3=_G3Residency(0)),
            second: _BlockRecord(g3=_G3Residency(2)),
        },
    )
    # _free_slots is the allocator's free list: recovery must rebuild it as the
    # complement of the adopted slots without disturbing the records themselves.
    assert tuple(g3._free_slots) == (1,)
    assert kvcr._core._block_record_map[first].g3 == _G3Residency(0)
    assert kvcr._core._block_record_map[second].g3 == _G3Residency(2)
    survivor = g3._descriptor(2)
    agent._file_data[(survivor.device_Id, survivor.addr)] = b"s" * page_size

    observed: list[tuple[BlockKey, int | None]] = []

    def observe(key: BlockKey, record: _BlockRecord) -> None:
        observed.append((key, None if record.g3 is None else record.g3.slot))

    g3.observe_residency(observe)

    # The first spill lands in the rebuilt free slot; the next can only land by
    # evicting a recovered block, and every move is reported before exposure.
    spilled, fresh, last = (
        BlockKey(b"spilled"),
        BlockKey(b"fresh"),
        BlockKey(b"last"),
    )
    primary_addr = ctypes.addressof(primary)
    for index, key in enumerate((spilled, fresh, last)):
        address = primary_addr + index * page_size
        assert _deposit(kvcr, key, address, page_size).success
    assert observed == [
        (spilled, 1),
        (first, None),
        (fresh, 0),
    ]
    assert kvcr.query((spilled, first, second)) == [
        (QueryStatus.FETCHABLE, CacheTier.G3),
        (QueryStatus.MISS, None),
        (QueryStatus.FETCHABLE, CacheTier.G3),
    ]

    deliver = kvcr.deliver(
        {second: _mem_descriptor(ctypes.addressof(destination), page_size)}
    )
    assert dict(_poll_until(kvcr, bool))[deliver][second].success
    assert destination.raw == b"s" * page_size


@pytest.mark.parametrize("slots", [(0, 0), (0, 4)])
def test_g3_recovery_rejects_invalid_slots(tmp_path, slots: tuple[int, int]) -> None:
    """Duplicate or out-of-range recovered slots are refused and none adopted."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    local = ctypes.create_string_buffer(page_size)
    kvcr = _new_g3_kvcr(tmp_path, local, g3_slot_count=4)
    g3 = kvcr._core._g3
    assert g3 is not None

    with pytest.raises(ValueError, match="invalid G3 recovery slots"):
        g3.adopt_recovery_slots(
            {
                BlockKey(b"first"): _BlockRecord(g3=_G3Residency(slots[0])),
                BlockKey(b"second"): _BlockRecord(g3=_G3Residency(slots[1])),
            }
        )

    assert kvcr._core._block_record_map == {}


def test_g3_spill_deliver_and_fill_reuse_existing_progress(tmp_path) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 2)
    primary.raw = b"a" * page_size + b"b" * page_size
    local = ctypes.create_string_buffer(page_size)
    destination = ctypes.create_string_buffer(page_size)
    first, second = BlockKey(b"first"), BlockKey(b"second")
    policy = _RecordingG3LRUPolicy()
    agent = _FakeG3Agent()
    kvcr = _new_g3_kvcr(tmp_path, local, agent=agent, policy=policy, telemetry=True)
    now = 0.0
    kvcr._core._clock = lambda: now

    assert agent.backends == {"MOCK": {"threads": "2"}}
    assert [mem_type for _, mem_type in agent.registrations] == [
        "FILE",
        "DRAM",
    ]
    assert _deposit(kvcr, first, ctypes.addressof(primary), page_size).success

    assert _deposit(
        kvcr,
        second,
        ctypes.addressof(primary) + page_size,
        page_size,
    ).success
    assert kvcr.query((first,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
    assert kvcr._core._block_record_map[first].local_dram is None
    local_deliver = kvcr.deliver(
        {second: _mem_descriptor(ctypes.addressof(destination), page_size)}
    )
    assert dict(_poll_until(kvcr, bool))[local_deliver][second].success
    now = 1.0
    deliver = kvcr.deliver(
        {first: _mem_descriptor(ctypes.addressof(destination), page_size)}
    )
    assert dict(_poll_until(kvcr, bool))[deliver][first].success
    assert destination.raw == b"a" * page_size
    assert kvcr.query((first,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]

    now = 2.0
    fetch = kvcr.fetch((first,))
    fetch_result = dict(_poll_until(kvcr, bool))[fetch][first]
    assert fetch_result.success and fetch_result.descriptor is not None
    assert ctypes.string_at(fetch_result.descriptor.addr, page_size) == b"a" * page_size
    record = kvcr._core._block_record_map[first]
    assert record.local_dram is not None and record.g3 is not None
    assert [
        (meta.access_count, meta.last_access)
        for meta, source in policy.scored
        if meta.block_key == first and source is CacheTier.G3
    ] == [(0, 0.0), (1, 1.0), (2, 2.0)]

    stats = kvcr.get_stats()
    assert isinstance(stats, FakeTelemetryStats)
    metrics = _metric_totals(stats)
    assert {
        ("gauge", STATE_METRIC, "local_g2_total_slots"): 1,
        ("gauge", STATE_METRIC, "local_g2_free_slots"): 0,
        ("gauge", STATE_METRIC, "local_g2_allocated_slots"): 1,
        ("gauge", STATE_METRIC, "local_g2_evictable_slots"): 0,
        ("gauge", STATE_METRIC, "g3_total_slots"): 2,
        ("gauge", STATE_METRIC, "g3_free_slots"): 0,
        ("gauge", STATE_METRIC, "g3_allocated_slots"): 2,
        ("gauge", STATE_METRIC, "g3_evictable_slots"): 2,
        ("gauge", STATE_METRIC, "g3_pending_stores"): 0,
        ("counter", TRANSFER_BLOCKS_METRIC, "local_fill"): 2,
        ("counter", TRANSFER_BLOCKS_METRIC, "local_deliver"): 1,
        ("counter", TRANSFER_BLOCKS_METRIC, "g3_store"): 2,
        ("counter", TRANSFER_BLOCKS_METRIC, "g3_fill"): 1,
        ("counter", TRANSFER_BLOCKS_METRIC, "g3_deliver"): 1,
        ("counter", TRANSFER_BYTES_METRIC, "local_fill"): page_size * 2,
        ("counter", TRANSFER_BYTES_METRIC, "local_deliver"): page_size,
        ("counter", TRANSFER_BYTES_METRIC, "g3_store"): page_size * 2,
        ("counter", TRANSFER_BYTES_METRIC, "g3_fill"): page_size,
        ("counter", TRANSFER_BYTES_METRIC, "g3_deliver"): page_size,
    }.items() <= metrics.items()
    for scope in (
        "local_fill",
        "local_deliver",
        "g3_store",
        "g3_fill",
        "g3_deliver",
    ):
        assert ("histogram", DURATION_METRIC, scope, "success") in metrics


def test_delayed_remote_fill_waves_use_private_transfer_ids(tmp_path) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 2)
    local = ctypes.create_string_buffer(page_size * 2)
    residents = (BlockKey(b"resident-0"), BlockKey(b"resident-1"))
    remotes = (BlockKey(b"remote-0"), BlockKey(b"remote-1"))
    agent = _FakeG3Agent()
    control = FakeBytesControl()
    kvcr = _new_g3_kvcr(
        tmp_path,
        local,
        agent=agent,
        slot_count=2,
        control=control,
        key_hint_adapter=_MatchingHintAdapter(),
        remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
    )

    for index, key in enumerate(residents):
        assert _deposit(
            kvcr,
            key,
            ctypes.addressof(primary) + index * page_size,
            page_size,
        ).success
    kvcr.submit_hint(remotes, src="tcp://source:1", request_id="req", hints="hint")
    fetch = kvcr.fetch(remotes, request_id="req")

    _poll_until(
        kvcr,
        lambda _: (
            len(
                [
                    message
                    for _, raw in control.sent
                    if (message := _decode_control_message(raw)).get("type")
                    == "start_write"
                ]
            )
            == 2
        ),
    )
    messages = [
        message
        for _, raw in control.sent
        if (message := _decode_control_message(raw)).get("type") == "start_write"
    ]
    fill_ids = {message["op_handle"] for message in messages}
    assert len(fill_ids) == 2 and all(fill_id < 0 for fill_id in fill_ids)
    assert fetch not in dict(kvcr.poll_completed())

    agent.notifs["source"] = [_write_done_notification(fill_id) for fill_id in fill_ids]
    result = dict(_poll_until(kvcr, lambda done: fetch in dict(done)))
    assert all(entry.success for entry in result[fetch].values())
    assert kvcr.query(residents) == [
        (QueryStatus.FETCHABLE, CacheTier.G3),
        (QueryStatus.FETCHABLE, CacheTier.G3),
    ]


def test_waiting_g3_fetch_source_is_not_evicted(tmp_path) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 4)
    local = ctypes.create_string_buffer(page_size * 2)
    first, second, third, fourth = (
        BlockKey(b"first"),
        BlockKey(b"second"),
        BlockKey(b"third"),
        BlockKey(b"fourth"),
    )
    kvcr = _new_g3_kvcr(tmp_path, local, slot_count=2)

    for index, key in enumerate((first, second, third, fourth)):
        assert _deposit(
            kvcr,
            key,
            ctypes.addressof(primary) + index * page_size,
            page_size,
        ).success
    assert kvcr.query((first, second)) == [
        (QueryStatus.FETCHABLE, CacheTier.G3),
        (QueryStatus.FETCHABLE, CacheTier.G3),
    ]

    fetch = kvcr.fetch((first,))
    result = dict(_poll_until(kvcr, lambda done: fetch in dict(done)))
    assert result[fetch][first].success
    assert kvcr.query((first,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert kvcr.query((second,)) == [(QueryStatus.MISS, None)]


def test_delayed_g3_fill_waves_use_private_transfer_ids(tmp_path) -> None:
    class _DelayedReadAgent(_FakeG3Agent):
        def __init__(self):
            super().__init__()
            self.allow_reads = False

        def transfer(self, handle):
            result = super().transfer(handle)
            operation = self.xfers[handle - 1][0]
            if operation == "READ" and self._xfer_backends[handle]:
                return "PROC"
            return result

        def check_xfer_state(self, handle):
            operation = self.xfers[handle - 1][0]
            if (
                operation == "READ"
                and self._xfer_backends[handle]
                and not self.allow_reads
            ):
                return "PROC"
            return "DONE"

    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 4)
    local = ctypes.create_string_buffer(page_size * 2)
    first, second, third, fourth = (
        BlockKey(b"first"),
        BlockKey(b"second"),
        BlockKey(b"third"),
        BlockKey(b"fourth"),
    )
    agent = _DelayedReadAgent()
    kvcr = _new_g3_kvcr(
        tmp_path,
        local,
        agent=agent,
        slot_count=2,
        g3_slot_count=4,
    )

    for index, key in enumerate((first, second, third, fourth)):
        assert _deposit(
            kvcr,
            key,
            ctypes.addressof(primary) + index * page_size,
            page_size,
        ).success
    fetch = kvcr.fetch((first, second))
    assert (
        len([op for op in kvcr._core._g3._active.values() if op.kind == "store"]) == 1
    )
    assert len(kvcr._core._local_dram._capacity_waiters) == 2

    _poll_until(
        kvcr,
        lambda _: (
            len([op for op in kvcr._core._g3._active.values() if op.kind == "fill"])
            == 2
        ),
    )
    fill_ids = {op.op_id for op in kvcr._core._g3._active.values() if op.kind == "fill"}
    assert len(fill_ids) == 2
    assert all(op_id[1] < 0 for op_id in fill_ids)
    assert fetch not in dict(kvcr.poll_completed())

    agent.allow_reads = True
    result = dict(_poll_until(kvcr, lambda done: fetch in dict(done)))
    assert all(entry.success for entry in result[fetch].values())


def test_failed_g3_spill_recovers_by_dropping_source(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="kvcr.policy")
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 2)
    local = ctypes.create_string_buffer(page_size)
    first, second = BlockKey(b"first"), BlockKey(b"second")
    agent = _FakeG3Agent()
    policy = _MoveLocalToG3Policy()
    kvcr = _new_g3_kvcr(tmp_path, local, agent=agent, policy=policy, telemetry=True)

    assert _deposit(kvcr, first, ctypes.addressof(primary), page_size).success
    assert _deposit(
        kvcr,
        second,
        ctypes.addressof(primary) + page_size,
        page_size,
    ).success
    agent._fail_file_writes = True

    fetch = kvcr.fetch((first,))
    result = dict(_poll_until(kvcr, lambda done: fetch in dict(done)))
    assert result[fetch][first].success
    assert kvcr.query((first,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert kvcr.query((second,)) == [(QueryStatus.MISS, None)]
    assert len(policy.failures) == 1
    meta, failure = policy.failures[0]
    assert meta.block_key == second
    assert failure.attempted == (PlacementAction.MOVE_TO, CacheTier.G3)
    assert failure.source is CacheTier.LOCAL_G2
    assert failure.reason == "transfer failed"
    assert failure.failure_count == 1
    assert any(
        "KVCR placement failed" in record.getMessage() for record in caplog.records
    )
    stats = kvcr.get_stats()
    assert isinstance(stats, FakeTelemetryStats)
    metrics = _metric_totals(stats)
    assert ("histogram", DURATION_METRIC, "g3_store", "failed") in metrics
    assert metrics[("counter", TRANSFER_BLOCKS_METRIC, "g3_store")] == 1


def test_g3_spill_waits_until_local_source_claim_is_released(tmp_path) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 2)
    local = ctypes.create_string_buffer(page_size)
    first, second = BlockKey(b"first"), BlockKey(b"second")
    kvcr = _new_g3_kvcr(tmp_path, local)

    assert _deposit(kvcr, first, ctypes.addressof(primary), page_size).success
    deposit = kvcr.deposit(
        {second: _mem_descriptor(ctypes.addressof(primary) + page_size, page_size)}
    )
    fetch = kvcr.fetch((first,))
    fetch_result = dict(_poll_until(kvcr, lambda done: bool(done)))[fetch][first]
    assert fetch_result.success and fetch_result.release_handle is not None

    _poll_until(
        kvcr,
        lambda _: kvcr._core._block_record_map[first].g3 is not None,
    )
    assert kvcr.query((first,)) == [(QueryStatus.HIT, CacheTier.LOCAL_G2)]
    assert deposit not in dict(kvcr.poll_completed())

    assert kvcr.release((fetch_result.release_handle,)) == [
        (fetch_result.release_handle, True)
    ]
    deposit_result = dict(_poll_until(kvcr, lambda done: deposit in dict(done)))[
        deposit
    ][second]
    assert deposit_result.success
    assert kvcr.query((first,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]


def test_fetch_falls_back_to_g3_while_a_local_fill_is_discarding(
    tmp_path,
) -> None:
    class _StuckReadAgent(_FakeG3Agent):
        """Keep a G3 read in flight so its fill is abandoned, not resolved."""

        def __init__(self):
            super().__init__()
            self.stuck = True

        def _is_stuck_read(self, handle):
            return (
                self.stuck
                and self.xfers[handle - 1][0] == "READ"
                and bool(self._xfer_backends.get(handle))
            )

        def transfer(self, handle):
            result = super().transfer(handle)
            return "PROC" if self._is_stuck_read(handle) else result

        def check_xfer_state(self, handle):
            return "PROC" if self._is_stuck_read(handle) else "DONE"

        def release_xfer_handle(self, handle):
            if self._is_stuck_read(handle):
                return False
            return super().release_xfer_handle(handle)

    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 2)
    primary.raw = b"a" * page_size + b"b" * page_size
    local = ctypes.create_string_buffer(page_size)
    first, second = BlockKey(b"first"), BlockKey(b"second")
    agent = _StuckReadAgent()
    policy = _MoveLocalToG3Policy()
    kvcr = _new_g3_kvcr(tmp_path, local, agent=agent, policy=policy)

    assert _deposit(kvcr, first, ctypes.addressof(primary), page_size).success
    assert _deposit(
        kvcr,
        second,
        ctypes.addressof(primary) + page_size,
        page_size,
    ).success
    _poll_until(kvcr, lambda _: kvcr._core._block_record_map[first].g3 is not None)
    policy.move = False  # later evictions drop instead of spilling

    now = [0.0]
    kvcr._core._clock = lambda: now[0]
    fetch = kvcr.fetch((first,))
    _poll_until(
        kvcr,
        lambda _: any(op.kind == "fill" for op in kvcr._core._g3._active.values()),
    )
    now[0] = 100.0
    result = dict(_poll_until(kvcr, lambda done: fetch in dict(done)))
    assert not result[fetch][first].success

    record = kvcr._core._block_record_map[first]
    assert record.local_dram.state is _LocalDramState.DISCARDING
    assert record.g3 is not None
    assert kvcr.query((first,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]

    # The abandoned fill still owns the slot, so the retry waits for it.
    retry = kvcr.fetch((first,))
    assert retry not in dict(kvcr.poll_completed())
    agent.stuck = False
    retry_result = dict(_poll_until(kvcr, lambda done: retry in dict(done)))[retry][
        first
    ]
    assert retry_result.success and retry_result.descriptor is not None
    assert ctypes.string_at(retry_result.descriptor.addr, page_size) == b"a" * page_size


def test_g3_deliver_rejects_a_destination_that_is_not_slot_sized(
    tmp_path,
) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 2)
    local = ctypes.create_string_buffer(page_size)
    destination = ctypes.create_string_buffer(page_size)
    first, second = BlockKey(b"first"), BlockKey(b"second")
    agent = _FakeG3Agent()
    kvcr = _new_g3_kvcr(tmp_path, local, agent=agent)

    assert _deposit(kvcr, first, ctypes.addressof(primary), page_size).success
    assert _deposit(
        kvcr,
        second,
        ctypes.addressof(primary) + page_size,
        page_size,
    ).success
    _poll_until(kvcr, lambda _: kvcr._core._block_record_map[first].g3 is not None)
    assert kvcr.query((first,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]

    reads = [operation for operation, *_ in agent.xfers if operation == "READ"]
    deliver = kvcr.deliver(
        {first: _mem_descriptor(ctypes.addressof(destination), page_size // 2)}
    )
    result = dict(_poll_until(kvcr, lambda done: deliver in dict(done)))
    assert not result[deliver][first].success
    # The undersized destination must never reach NIXL as a whole-slot read.
    assert [operation for operation, *_ in agent.xfers if operation == "READ"] == reads
    assert kvcr.query((first,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]


def test_closing_an_unfinished_spill_releases_its_capacity_reservation(
    tmp_path,
) -> None:
    class _StuckWriteAgent(_FakeG3Agent):
        def _is_file_write(self, handle):
            return self.xfers[handle - 1][0] == "WRITE" and bool(
                self._xfer_backends.get(handle)
            )

        def transfer(self, handle):
            result = super().transfer(handle)
            return "PROC" if self._is_file_write(handle) else result

        def check_xfer_state(self, handle):
            return "PROC" if self._is_file_write(handle) else "DONE"

    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 2)
    local = ctypes.create_string_buffer(page_size)
    first, second = BlockKey(b"first"), BlockKey(b"second")
    kvcr = _new_g3_kvcr(tmp_path, local, agent=_StuckWriteAgent())

    assert _deposit(kvcr, first, ctypes.addressof(primary), page_size).success
    kvcr.deposit(
        {second: _mem_descriptor(ctypes.addressof(primary) + page_size, page_size)}
    )
    local_dram = kvcr._core._local_dram
    _poll_until(kvcr, lambda _: local_dram._capacity_eviction_key == first)

    _OPEN_KVCRS.remove(kvcr)
    kvcr.close()
    # Nothing will finish the spill now, so nothing may stay queued behind it.
    assert local_dram._capacity_eviction_key is None


def test_waiting_g3_fetch_obeys_original_deadline(tmp_path) -> None:
    page_size = os.sysconf("SC_PAGE_SIZE")
    primary = ctypes.create_string_buffer(page_size * 2)
    local = ctypes.create_string_buffer(page_size)
    first, second = BlockKey(b"first"), BlockKey(b"second")
    agent = _FakeG3Agent()
    kvcr = _new_g3_kvcr(tmp_path, local, agent=agent)

    assert _deposit(kvcr, first, ctypes.addressof(primary), page_size).success
    assert _deposit(
        kvcr,
        second,
        ctypes.addressof(primary) + page_size,
        page_size,
    ).success
    now = [0.0]
    kvcr._core._clock = lambda: now[0]
    agent.state = "PROC"
    fetch = kvcr.fetch((first,))
    now[0] = 2.0

    result = dict(kvcr.poll_completed())
    assert fetch in result and not result[fetch][first].success
    _poll_until(kvcr, lambda _: not _has_outstanding_operations(kvcr))
    assert kvcr.query((first,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
    assert kvcr.query((second,)) == [(QueryStatus.MISS, None)]
