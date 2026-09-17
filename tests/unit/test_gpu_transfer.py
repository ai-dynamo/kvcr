# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real NIXL transfers between framework GPU memory and KVCR-owned DRAM.

Skipped without CUDA and torch. Runs one UCX agent in-process, so it proves
the local transfer path (GPU source -> KVCR DRAM through deposit, KVCR DRAM ->
GPU destination through deliver), the per-layer multi-allocation case, and that
completion means the bytes are visible after the consuming stream synchronizes.
"""

import socket
import time
import uuid

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - hardware gate
    pytest.skip("CUDA is required", allow_module_level=True)
pytest.importorskip("nixl")

from kvcr import KVCR, KVCRBindings  # noqa: E402
from kvcr.config import (  # noqa: E402
    FrameworkMemoryRegion,
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramOptions,
)
from kvcr.types import BlockKey, MemDescriptor, OpEntryStatus  # noqa: E402

_TIMEOUT_S = 20.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _NoPinning:
    def request_pin(self, keys):
        return 0

    def poll_pin_results(self):
        return ()

    def release_pin(self, handle):
        return False


def _region(tensor: torch.Tensor) -> FrameworkMemoryRegion:
    storage = tensor.untyped_storage()
    return FrameworkMemoryRegion(
        address=storage.data_ptr(),
        length=storage.nbytes(),
        mem_type="VRAM" if tensor.is_cuda else "DRAM",
        device_id=tensor.device.index if tensor.is_cuda else 0,
        owner=tensor,
    )


def _descriptors(
    agent_name: str, tensor: torch.Tensor, spans: list[tuple[int, int, str]]
):
    return [
        MemDescriptor(
            end_point_name=agent_name,
            mem_type="VRAM" if tensor.is_cuda else "DRAM",
            addr=tensor.data_ptr() + offset,
            size=size,
            device_Id=tensor.device.index if tensor.is_cuda else 0,
            info=pool,
        )
        for offset, size, pool in spans
    ]


def _wait(kvcr: KVCR, handle: int):
    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline:
        for done, entries in kvcr.poll_completed():
            if done == handle:
                return entries
        time.sleep(0.001)
    raise AssertionError("KVCR operation did not complete")


def _make_kvcr(name: str, pool_layouts, local_pool: torch.Tensor, regions):
    pool_address = local_pool.data_ptr()
    offset = 0
    pools = []
    span_bytes = sum(size for _, size in pool_layouts)
    slots = local_pool.numel() // span_bytes
    for pool_name, size in pool_layouts:
        pools.append((pool_name, pool_address + offset, size * slots))
        offset += size * slots
    return KVCR(
        KVCRConfig(
            nixl_agent_name=name,
            pool_layouts=list(pool_layouts),
            nixl_listen_port=_free_port(),
            operation_timeout_ms=10_000,
            abandon_timeout_ms=20_000,
        ),
        KVCRBindings(
            request_pin=_NoPinning().request_pin,
            poll_pin_results=_NoPinning().poll_pin_results,
            release_pin=_NoPinning().release_pin,
        ),
        KVCRBackendConfigs(
            framework_regions=tuple(regions),
            local_dram=LocalDramOptions(pools, backend="UCX"),
        ),
    )


def test_gpu_deposit_then_deliver_round_trips_bytes() -> None:
    # Two "layer" allocations per page, each a separate GPU tensor, so the
    # object is an ordered two-span descriptor list over two registrations.
    name = f"kvcr-gpu-{uuid.uuid4().hex[:8]}"
    span = 4096
    layer_a = torch.arange(span * 2, dtype=torch.uint8, device="cuda").reshape(2, span)
    layer_b = (
        torch.arange(span * 2, dtype=torch.uint8, device="cuda")
        .flip(0)
        .reshape(2, span)
    )
    restore_a = torch.zeros_like(layer_a)
    restore_b = torch.zeros_like(layer_b)
    torch.cuda.synchronize()
    local_pool = torch.empty(span * 2 * 4, dtype=torch.uint8, pin_memory=True)
    layouts = [("a", span), ("b", span)]
    kvcr = _make_kvcr(
        name,
        layouts,
        local_pool,
        [_region(t) for t in (layer_a, layer_b, restore_a, restore_b)],
    )
    try:
        keys = [BlockKey(b"page0"), BlockKey(b"page1")]
        deposit = kvcr.deposit(
            {
                key: _descriptors(name, layer_a, [(page * span, span, "a")])
                + _descriptors(name, layer_b, [(page * span, span, "b")])
                for page, key in enumerate(keys)
            }
        )
        entries = _wait(kvcr, deposit)
        assert {k: e.status for k, e in entries.items()} == {
            k: OpEntryStatus.SUCCESS for k in keys
        }

        deliver = kvcr.deliver(
            {
                key: _descriptors(name, restore_a, [(page * span, span, "a")])
                + _descriptors(name, restore_b, [(page * span, span, "b")])
                for page, key in enumerate(keys)
            }
        )
        entries = _wait(kvcr, deliver)
        assert all(e.success for e in entries.values())
        torch.cuda.synchronize()
        assert torch.equal(restore_a, layer_a)
        assert torch.equal(restore_b, layer_b)
        # Bytes did land in KVCR-owned DRAM, not only in the destination.
        host_view = local_pool[: span * 4].view(2, 2, span)
        assert torch.equal(host_view[0].cpu(), layer_a.cpu())
    finally:
        kvcr.close()


def test_mixed_dram_and_gpu_sources_in_one_deposit() -> None:
    name = f"kvcr-gpu-{uuid.uuid4().hex[:8]}"
    span = 2048
    gpu_src = torch.full((span,), 7, dtype=torch.uint8, device="cuda")
    host_src = torch.full((span,), 9, dtype=torch.uint8, pin_memory=True)
    gpu_dst = torch.zeros(span * 2, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    local_pool = torch.empty(span * 4, dtype=torch.uint8, pin_memory=True)
    kvcr = _make_kvcr(
        name,
        [("", span)],
        local_pool,
        [_region(gpu_src), _region(host_src), _region(gpu_dst)],
    )
    try:
        deposit = kvcr.deposit(
            {
                BlockKey(b"gpu"): _descriptors(name, gpu_src, [(0, span, "")]),
                BlockKey(b"host"): _descriptors(name, host_src, [(0, span, "")]),
            }
        )
        assert all(e.success for e in _wait(kvcr, deposit).values())
        deliver = kvcr.deliver(
            {
                BlockKey(b"gpu"): _descriptors(name, gpu_dst, [(0, span, "")]),
                BlockKey(b"host"): _descriptors(name, gpu_dst, [(span, span, "")]),
            }
        )
        assert all(e.success for e in _wait(kvcr, deliver).values())
        torch.cuda.synchronize()
        assert int(gpu_dst[:span].min()) == 7 and int(gpu_dst[:span].max()) == 7
        assert int(gpu_dst[span:].min()) == 9 and int(gpu_dst[span:].max()) == 9
    finally:
        kvcr.close()


def test_unregistered_gpu_source_fails_the_entry_not_the_core() -> None:
    # A descriptor outside every registered region must surface as a failed
    # entry (or a raised submission), never as a silent success.
    name = f"kvcr-gpu-{uuid.uuid4().hex[:8]}"
    span = 1024
    registered = torch.ones(span, dtype=torch.uint8, device="cuda")
    stray = torch.ones(span, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    local_pool = torch.empty(span * 2, dtype=torch.uint8, pin_memory=True)
    kvcr = _make_kvcr(name, [("", span)], local_pool, [_region(registered)])
    try:
        deposit = kvcr.deposit(
            {BlockKey(b"stray"): _descriptors(name, stray, [(0, span, "")])}
        )
        entries = _wait(kvcr, deposit)
        assert not entries[BlockKey(b"stray")].success
    finally:
        kvcr.close()


def test_startup_fails_cleanly_on_bogus_gpu_region() -> None:
    name = f"kvcr-gpu-{uuid.uuid4().hex[:8]}"
    span = 1024
    local_pool = torch.empty(span * 2, dtype=torch.uint8, pin_memory=True)
    bogus = FrameworkMemoryRegion(
        address=0x10, length=span, mem_type="VRAM", device_id=0
    )
    with pytest.raises(Exception):
        _make_kvcr(name, [("", span)], local_pool, [bogus])
    # The same agent name and a valid region start afterwards: nothing leaked
    # a registration or a listener that the retry would trip over.
    good = torch.ones(span, dtype=torch.uint8, device="cuda")
    kvcr = _make_kvcr(name, [("", span)], local_pool, [_region(good)])
    kvcr.close()
