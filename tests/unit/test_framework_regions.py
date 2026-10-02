# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Typed framework memory regions: registration, validation, and copy batching."""

import ctypes
import gc
import weakref

import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _new_kvcr,
    _poll_until,
)

from kvcr.config import (
    FrameworkDramInput,
    FrameworkMemoryRegion,
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramOptions,
    resolve_framework_regions,
)
from kvcr.local_dram import _LocalCopyOp, _LocalDramState
from kvcr.types import BlockKey, CacheTier, MemDescriptor, OpEntryStatus, QueryStatus

_BLOCK = 16


def _descriptor(addr: int, *, mem_type: str = "DRAM", device_id: int = 0, info=""):
    return MemDescriptor("target", mem_type, addr, _BLOCK, device_id, info)


def _local_kvcr(agent: FakeNixlAgent, pool: ctypes.Array, **regions):
    return _new_kvcr(
        agent,
        FakePrimaryPinning(),
        FakeBytesControl(),
        KVCRConfig(nixl_agent_name="target", pool_layouts=[("", _BLOCK)]),
        local_dram=LocalDramOptions([("", ctypes.addressof(pool), len(pool))]),
        **regions,
    )


def test_regions_register_once_per_memory_type_and_device() -> None:
    # Two views of one GPU allocation collapse; a second device gets its own
    # registration; the DRAM group carries the framework and pool regions.
    pool = ctypes.create_string_buffer(_BLOCK * 4)
    agent = FakeNixlAgent()
    owner = object()
    regions = (
        FrameworkMemoryRegion(0x1000, 0x100, "VRAM", 0, owner=owner),
        FrameworkMemoryRegion(0x1000, 0x100, "VRAM", 0),
        FrameworkMemoryRegion(0x2000, 0x100, "VRAM", 1),
        FrameworkMemoryRegion(0x3000, 0x40, "DRAM", 0),
    )
    kvcr = _local_kvcr(agent, pool, framework_regions=regions)

    assert agent.registrations == [
        ([(0x1000, 0x100, 0, "")], "VRAM"),
        ([(0x2000, 0x100, 1, "")], "VRAM"),
        (
            [(0x3000, 0x40, 0, ""), (ctypes.addressof(pool), len(pool), 0, "")],
            "DRAM",
        ),
    ]
    kvcr.close()
    assert sorted(agent.deregistered) == [1, 2, 3]


def test_legacy_framework_dram_still_registers_as_dram() -> None:
    pool = ctypes.create_string_buffer(_BLOCK * 4)
    agent = FakeNixlAgent()
    kvcr = _local_kvcr(agent, pool, framework_dram=FrameworkDramInput(0x3000, 0x40))

    assert agent.registrations == [
        (
            [(0x3000, 0x40, 0, ""), (ctypes.addressof(pool), len(pool), 0, "")],
            "DRAM",
        ),
    ]
    kvcr.close()


def test_legacy_and_typed_regions_together_are_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        resolve_framework_regions(
            KVCRBackendConfigs(
                framework_dram=FrameworkDramInput(0x1000, 0x10),
                framework_regions=(FrameworkMemoryRegion(0x2000, 0x10),),
            )
        )


def test_overlapping_regions_on_one_endpoint_are_refused() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        resolve_framework_regions(
            KVCRBackendConfigs(
                framework_regions=(
                    FrameworkMemoryRegion(0x1000, 0x100, "VRAM", 0),
                    FrameworkMemoryRegion(0x1080, 0x100, "VRAM", 0),
                )
            )
        )
    # The same span on another device is a different allocation.
    resolved = resolve_framework_regions(
        KVCRBackendConfigs(
            framework_regions=(
                FrameworkMemoryRegion(0x1000, 0x100, "VRAM", 0),
                FrameworkMemoryRegion(0x1080, 0x100, "VRAM", 1),
            )
        )
    )
    assert len(resolved) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"address": 0, "length": 1},
        {"address": 1, "length": 0},
        {"address": 1, "length": 1, "mem_type": "FILE"},
        {"address": 1, "length": 1, "device_id": -1},
    ],
)
def test_invalid_regions_are_refused(kwargs) -> None:
    with pytest.raises(ValueError):
        FrameworkMemoryRegion(**kwargs)


def test_region_owner_is_retained_until_close() -> None:
    pool = ctypes.create_string_buffer(_BLOCK * 4)

    class Owner:
        pass

    owner = Owner()
    alive = weakref.ref(owner)
    kvcr = _local_kvcr(
        FakeNixlAgent(),
        pool,
        framework_regions=(
            FrameworkMemoryRegion(0x1000, 0x100, "VRAM", 0, owner=owner),
        ),
    )
    del owner
    gc.collect()
    assert alive() is not None, "KVCR must pin the owner while registered"
    kvcr.close()
    gc.collect()
    assert alive() is None


def test_startup_registration_failure_deregisters_earlier_groups() -> None:
    pool = ctypes.create_string_buffer(_BLOCK * 4)

    class RejectVram(FakeNixlAgent):
        def register_memory(self, descs, mem_type="DRAM"):
            if mem_type == "VRAM":
                raise RuntimeError("no CUDA context")
            return super().register_memory(descs, mem_type)

    agent = RejectVram()
    with pytest.raises(RuntimeError, match="no CUDA context"):
        _local_kvcr(
            agent,
            pool,
            framework_regions=(
                FrameworkMemoryRegion(0x3000, 0x40, "DRAM", 0),
                FrameworkMemoryRegion(0x1000, 0x100, "VRAM", 0),
            ),
        )
    # The DRAM group registered first and was released on the failed start.
    assert agent.registrations == [
        (
            [(0x3000, 0x40, 0, ""), (ctypes.addressof(pool), len(pool), 0, "")],
            "DRAM",
        ),
    ]
    assert agent.deregistered == [1]


def test_block_descriptors_must_share_one_endpoint() -> None:
    pool = ctypes.create_string_buffer(_BLOCK * 4)
    kvcr = _local_kvcr(FakeNixlAgent(), pool)
    mixed = [_descriptor(0x1000, mem_type="VRAM"), _descriptor(0x2000)]
    with pytest.raises(ValueError, match="share one memory type and device"):
        kvcr.deposit({BlockKey(b"k"): mixed})
    with pytest.raises(ValueError, match="share one memory type and device"):
        kvcr.deliver({BlockKey(b"k"): mixed})


def _submitted_copies(agent: FakeNixlAgent) -> list[tuple[str, list, list]]:
    return [(op, local, remote) for op, local, _, remote, _, _ in agent.xfers]


def test_deposit_batches_copies_per_source_endpoint() -> None:
    # Sources on two GPUs and host DRAM: three transfers, each a single memory
    # type, in first-seen order, and each block still completes.
    pool = ctypes.create_string_buffer(_BLOCK * 8)
    payloads = [ctypes.create_string_buffer(bytes([i]) * _BLOCK) for i in range(4)]
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _local_kvcr(agent, pool)
    blocks = {
        BlockKey(b"g0a"): [
            _descriptor(ctypes.addressof(payloads[0]), mem_type="VRAM", device_id=0)
        ],
        BlockKey(b"h0"): [_descriptor(ctypes.addressof(payloads[1]))],
        BlockKey(b"g1"): [
            _descriptor(ctypes.addressof(payloads[2]), mem_type="VRAM", device_id=1)
        ],
        BlockKey(b"g0b"): [
            _descriptor(ctypes.addressof(payloads[3]), mem_type="VRAM", device_id=0)
        ],
    }
    op_handle = kvcr.deposit(blocks)
    completed = dict(_poll_until(kvcr, lambda done: len(done) == 1))[op_handle]

    assert {key: entry.status for key, entry in completed.items()} == {
        key: OpEntryStatus.SUCCESS for key in blocks
    }
    copies = _submitted_copies(agent)
    assert [len(local) for _, local, _ in copies] == [2, 1, 1]
    # g0a and g0b share device 0 and travel together, in submission order.
    assert [local[0][0] for _, local, _ in copies] == [
        ctypes.addressof(payloads[0]),
        ctypes.addressof(payloads[1]),
        ctypes.addressof(payloads[2]),
    ]
    assert copies[0][1][1][0] == ctypes.addressof(payloads[3])
    assert [local[0][2] for _, local, _ in copies] == [0, 0, 1]
    kvcr.close()


def test_deliver_batches_copies_per_destination_endpoint() -> None:
    pool = ctypes.create_string_buffer(_BLOCK * 8)
    payloads = [ctypes.create_string_buffer(bytes([i + 1]) * _BLOCK) for i in range(2)]
    targets = [ctypes.create_string_buffer(_BLOCK) for _ in range(2)]
    agent = FakeNixlAgent()
    agent.state = "DONE"
    kvcr = _local_kvcr(agent, pool)
    keys = (BlockKey(b"a"), BlockKey(b"b"))
    deposit = kvcr.deposit(
        {key: [_descriptor(ctypes.addressof(buf))] for key, buf in zip(keys, payloads)}
    )
    _poll_until(kvcr, lambda done: any(handle == deposit for handle, _ in done))

    deliver = kvcr.deliver(
        {
            keys[0]: [_descriptor(ctypes.addressof(targets[0]), mem_type="VRAM")],
            keys[1]: [_descriptor(ctypes.addressof(targets[1]))],
        }
    )
    result = dict(
        _poll_until(kvcr, lambda done: any(handle == deliver for handle, _ in done))
    )[deliver]
    assert all(entry.success for entry in result.values())
    delivery_copies = _submitted_copies(agent)[1:]
    assert len(delivery_copies) == 2
    assert [remote[0][0] for _, _, remote in delivery_copies] == [
        ctypes.addressof(targets[0]),
        ctypes.addressof(targets[1]),
    ]
    assert targets[0].raw[:_BLOCK] == payloads[0].raw[:_BLOCK]
    assert targets[1].raw[:_BLOCK] == payloads[1].raw[:_BLOCK]
    kvcr.close()


def test_pending_copy_is_cancelled_and_discarded_on_close() -> None:
    # A deposit whose NIXL transfer never reports DONE: close() cancels it,
    # the block never becomes resident, and the slot returns to the free list.
    pool = ctypes.create_string_buffer(_BLOCK * 2)
    payload = ctypes.create_string_buffer(b"z" * _BLOCK)
    agent = FakeNixlAgent()
    agent.state = "PROC"
    kvcr = _local_kvcr(agent, pool)
    key = BlockKey(b"stuck")
    kvcr.deposit({key: [_descriptor(ctypes.addressof(payload), mem_type="VRAM")]})
    # Let progress pick the copy up and post the transfer.
    _poll_until(kvcr, lambda _done: bool(agent.transfers), timeout=1)
    assert isinstance(
        next(iter(kvcr._core._progress._in_flight_ops.values())), _LocalCopyOp
    )

    kvcr.close()

    assert kvcr._core.is_quiescent()
    assert agent.released_xfers == [1]
    # The fill never committed: the block must not read as resident data.
    record = kvcr._core._block_record_map.get(key)
    assert (
        record is None
        or record.local_dram is None
        or (record.local_dram.state is not _LocalDramState.READY)
    )
    assert kvcr.query((key,)) == [(QueryStatus.MISS, None)] or kvcr.query((key,)) == [
        (QueryStatus.FETCHING, CacheTier.LOCAL_G2)
    ]
