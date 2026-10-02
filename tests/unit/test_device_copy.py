"""The CUDA copy engine and its use by local copy ops, driven by a fake runtime."""

import ctypes
from unittest.mock import patch

import numpy as np

from kvcr.config import FrameworkMemoryRegion
from kvcr.device_copy import (
    _ERROR_NOT_READY,
    _MIN_SPANS_PER_STREAM,
    _STREAMS_PER_DEVICE,
    CudaError,
    CudaRuntime,
    DeviceCopyEngine,
    DeviceCopyRequest,
    create_device_copy_engine,
)
from kvcr.local_dram import _LocalCopyOp
from kvcr.types import BlockKey, MemDescriptor

_ILLEGAL_ADDRESS = 700
_INVALID_VALUE = 1


class FakeRuntime:
    """Queue copies per stream and apply them when the recorded event fires.

    Events report not-ready ``latency`` times before completing, so tests can
    observe the window in which a DMA is still in flight.
    """

    has_memcpy_batch = True

    def __init__(self, *, latency: int = 1, batch_error: int | None = None) -> None:
        self.latency = latency
        self.batch_error = batch_error
        self.devices: list[int] = []
        self.calls: list[tuple[str, int]] = []
        self.destroyed_streams: list[int] = []
        self.destroyed_events: list[int] = []
        self._streams = 0
        self._events = 0
        self._queued: dict[int, list[tuple[int, int, int]]] = {}
        self._event_copies: dict[int, list[tuple[int, int, int]]] = {}
        self._event_remaining: dict[int, int] = {}

    def error_string(self, code: int) -> str:
        return f"fake error {code}"

    def check(self, code: int, what: str) -> None:
        if code != 0:
            raise CudaError(what, code, self.error_string(code))

    def device_count(self) -> int:
        return 2

    def set_device(self, device_id: int) -> None:
        self.devices.append(device_id)

    def stream_create(self) -> int:
        self._streams += 1
        return 100 + self._streams

    def stream_destroy(self, stream: int) -> int:
        self.destroyed_streams.append(stream)
        return 0

    def event_create(self) -> int:
        self._events += 1
        return 200 + self._events

    def event_destroy(self, event: int) -> int:
        self.destroyed_events.append(event)
        return 0

    def event_record(self, event: int, stream: int) -> None:
        self._event_copies[event] = self._queued.pop(stream, [])
        self._event_remaining[event] = self.latency

    def event_query(self, event: int) -> int:
        if self._event_remaining[event] > 0:
            self._event_remaining[event] -= 1
            return _ERROR_NOT_READY
        for dst, src, size in self._event_copies.pop(event, []):
            ctypes.memmove(dst, src, size)
        return 0

    def memcpy_async(self, dst: int, src: int, size: int, stream: int) -> int:
        self.calls.append(("single", size))
        self._queued.setdefault(stream, []).append((dst, src, size))
        return 0

    def memcpy_batch_async(
        self, dsts: np.ndarray, srcs: np.ndarray, sizes: np.ndarray, stream: int
    ) -> int:
        assert dsts.dtype == np.uint64 and dsts.flags.c_contiguous
        self.calls.append(("batch", len(dsts)))
        if self.batch_error is not None:
            return self.batch_error
        self._queued.setdefault(stream, []).extend(
            zip(dsts.tolist(), srcs.tolist(), sizes.tolist())
        )
        return 0


_SPAN = 16


def _buffers(count: int, fill: int):
    return [
        ctypes.create_string_buffer(bytes([fill]) * _SPAN, _SPAN) for _ in range(count)
    ]


def _descriptor(
    buffer, mem_type: str, device_id: int = 0, size: int = _SPAN
) -> MemDescriptor:
    return MemDescriptor(
        end_point_name="agent",
        mem_type=mem_type,
        addr=ctypes.addressof(buffer),
        size=size,
        device_Id=device_id,
        info="",
    )


def _region_over(buffers, device_id: int = 0) -> FrameworkMemoryRegion:
    start = min(ctypes.addressof(buffer) for buffer in buffers)
    end = max(ctypes.addressof(buffer) for buffer in buffers) + _SPAN
    return FrameworkMemoryRegion(start, end - start, "VRAM", device_id)


def _engine(runtime: FakeRuntime, *buffers) -> DeviceCopyEngine:
    return DeviceCopyEngine(runtime, (_region_over(buffers),))


def _request(engine: DeviceCopyEngine, sources, targets) -> DeviceCopyRequest | str:
    """Build a request the way local DRAM does: framework side validated first."""
    source_arrays = engine.span_arrays(sources)
    if isinstance(source_arrays, str):
        return source_arrays
    target_arrays = engine.span_arrays(targets)
    if isinstance(target_arrays, str):
        return target_arrays
    device = (
        sources[0].device_Id if sources[0].mem_type == "VRAM" else targets[0].device_Id
    )
    return DeviceCopyEngine.request(device, *target_arrays, *source_arrays)


def _submit(engine: DeviceCopyEngine, sources, targets):
    request = _request(engine, sources, targets)
    assert not isinstance(request, str), request
    return engine.submit(request)


def _drain(engine: DeviceCopyEngine, handle, limit: int = 10):
    for _ in range(limit):
        result = engine.poll(handle)
        if result is not None:
            return result
    raise AssertionError("copy never completed")


def test_batch_copy_lands_when_its_event_completes() -> None:
    runtime = FakeRuntime(latency=1)
    gpu = _buffers(2, 7)
    dram = _buffers(2, 0)
    engine = _engine(runtime, *gpu)
    handle = _submit(
        engine,
        [_descriptor(b, "VRAM") for b in gpu],
        [_descriptor(b, "DRAM") for b in dram],
    )
    assert handle.error is None and handle.device_id == 0
    assert handle.byte_count == 2 * _SPAN
    assert engine.poll(handle) is None
    assert dram[0].raw == bytes(_SPAN), "DMA must not be observable before the event"
    assert engine.poll(handle) is True
    assert all(buffer.raw == bytes([7]) * _SPAN for buffer in dram)
    assert runtime.calls == [("batch", 2)]
    assert runtime.devices == [0]
    assert (engine.submitted, engine.completed, engine.failed) == (1, 1, 0)
    assert engine.bytes_completed == 2 * _SPAN


def test_large_batches_split_across_streams() -> None:
    runtime = FakeRuntime(latency=1)
    # The fake spans are tiny; let the span count alone decide the streams.
    patcher = patch("kvcr.device_copy._MIN_BYTES_PER_STREAM", 1)
    patcher.start()
    count = _MIN_SPANS_PER_STREAM * _STREAMS_PER_DEVICE
    gpu = _buffers(count, 3)
    dram = _buffers(count, 0)
    engine = _engine(runtime, *gpu)
    handle = _submit(
        engine,
        [_descriptor(b, "VRAM") for b in gpu],
        [_descriptor(b, "DRAM") for b in dram],
    )
    assert len(handle.events) == _STREAMS_PER_DEVICE
    assert runtime.calls == [("batch", _MIN_SPANS_PER_STREAM)] * _STREAMS_PER_DEVICE
    assert engine.poll(handle) is None
    assert _drain(engine, handle) is True
    assert all(buffer.raw == bytes([3]) * _SPAN for buffer in dram)
    assert runtime._streams == _STREAMS_PER_DEVICE
    # A smaller follow-up batch reuses one existing stream instead of creating more.
    small = _submit(
        engine, [_descriptor(gpu[0], "VRAM")], [_descriptor(dram[0], "DRAM")]
    )
    assert _drain(engine, small) is True
    assert runtime._streams == _STREAMS_PER_DEVICE

    patcher.stop()


def test_deliver_direction_uses_destination_device() -> None:
    runtime = FakeRuntime(latency=0)
    gpu = _buffers(1, 0)
    dram = _buffers(1, 3)
    engine = DeviceCopyEngine(runtime, (_region_over(gpu, device_id=1),))
    handle = _submit(
        engine,
        [_descriptor(dram[0], "DRAM")],
        [_descriptor(gpu[0], "VRAM", device_id=1)],
    )
    assert handle.device_id == 1
    assert _drain(engine, handle) is True
    assert gpu[0].raw == bytes([3]) * _SPAN
    assert runtime.devices == [1]


def test_unregistered_vram_span_is_rejected_before_touching_cuda() -> None:
    runtime = FakeRuntime()
    gpu = _buffers(1, 1)
    stray = _buffers(1, 2)
    dram = _buffers(1, 0)
    engine = _engine(runtime, *gpu)
    assert engine.covers(_descriptor(gpu[0], "VRAM"))
    assert not engine.covers(_descriptor(stray[0], "VRAM"))
    assert not engine.covers(_descriptor(gpu[0], "VRAM", size=_SPAN + 1))
    assert not engine.covers(_descriptor(gpu[0], "VRAM", device_id=1))
    problem = _request(
        engine, [_descriptor(stray[0], "VRAM")], [_descriptor(dram[0], "DRAM")]
    )
    assert isinstance(problem, str) and "not a registered framework region" in problem
    assert runtime.devices == [] and runtime.calls == []
    assert dram[0].raw == bytes(_SPAN)


def test_span_size_mismatch_is_rejected() -> None:
    runtime = FakeRuntime()
    gpu = _buffers(1, 1)
    dram = _buffers(1, 0)
    engine = _engine(runtime, *gpu)
    problem = _request(
        engine,
        [_descriptor(gpu[0], "VRAM")],
        [_descriptor(dram[0], "DRAM", size=_SPAN // 2)],
    )
    assert isinstance(problem, str) and "span sizes differ" in problem
    assert isinstance(
        DeviceCopyEngine.request(
            0,
            np.zeros(2, dtype=np.uint64),
            np.full(2, _SPAN, dtype=np.uint64),
            np.zeros(1, dtype=np.uint64),
            np.full(1, _SPAN, dtype=np.uint64),
        ),
        str,
    )


def test_mixed_devices_in_one_list_are_rejected() -> None:
    runtime = FakeRuntime()
    gpu = _buffers(2, 1)
    engine = DeviceCopyEngine(
        runtime, (_region_over(gpu[:1]), _region_over(gpu[1:], device_id=1))
    )
    problem = engine.span_arrays(
        [_descriptor(gpu[0], "VRAM"), _descriptor(gpu[1], "VRAM", device_id=1)]
    )
    assert isinstance(problem, str) and "cannot mix" in problem
    assert isinstance(engine.span_arrays([]), str)


def test_rejected_batch_falls_back_to_per_span_copies() -> None:
    runtime = FakeRuntime(latency=0, batch_error=_INVALID_VALUE)
    gpu = _buffers(2, 5)
    dram = _buffers(2, 0)
    engine = _engine(runtime, *gpu)
    sources = [_descriptor(b, "VRAM") for b in gpu]
    targets = [_descriptor(b, "DRAM") for b in dram]
    assert _drain(engine, _submit(engine, sources, targets)) is True
    assert runtime.calls == [("batch", 2), ("single", _SPAN), ("single", _SPAN)]
    assert not engine.uses_batch_memcpy
    assert all(buffer.raw == bytes([5]) * _SPAN for buffer in dram)
    assert _drain(engine, _submit(engine, sources, targets)) is True
    assert runtime.calls[3:] == [("single", _SPAN), ("single", _SPAN)]


def test_hard_batch_error_fails_the_handle_but_still_records_an_event() -> None:
    runtime = FakeRuntime(latency=1, batch_error=_ILLEGAL_ADDRESS)
    gpu = _buffers(2, 5)
    dram = _buffers(2, 0)
    engine = _engine(runtime, *gpu)
    handle = _submit(
        engine,
        [_descriptor(b, "VRAM") for b in gpu],
        [_descriptor(b, "DRAM") for b in dram],
    )
    assert "cudaMemcpyBatchAsync" in (handle.error or "")
    assert len(handle.events) == 1, "in-flight spans keep their slots until the event"
    assert engine.poll(handle) is None
    assert engine.poll(handle) is False
    assert engine.uses_batch_memcpy, "a hard error is not a capability signal"
    assert engine.failed == 1


def test_close_destroys_streams_and_recycled_events() -> None:
    runtime = FakeRuntime(latency=0)
    gpu = _buffers(1, 1)
    dram = _buffers(1, 0)
    engine = _engine(runtime, *gpu)
    sources, targets = [_descriptor(gpu[0], "VRAM")], [_descriptor(dram[0], "DRAM")]
    assert _drain(engine, _submit(engine, sources, targets)) is True
    assert _drain(engine, _submit(engine, sources, targets)) is True
    assert runtime._events == 1, "a completed event is recycled, not recreated"
    engine.close()
    assert runtime.destroyed_streams == [101]
    assert runtime.destroyed_events == [201]


def _copy_op(engine: DeviceCopyEngine, sources, targets, clock, deadline: float):
    request = _request(engine, sources, targets)
    if isinstance(request, str):
        byte_count, request, error = _SPAN, None, request
    else:
        byte_count, error = request.byte_count, None
    return _LocalCopyOp(
        op_id=("local_copy", 0),
        keys={BlockKey(b"k")},
        deliver_op_id=None,
        ordered_keys=(BlockKey(b"k"),),
        local_slots=((("", 0),),),
        byte_count=byte_count,
        deadline=deadline,
        backend="UCX",
        clock=clock,
        started_at=None,
        device_copy=engine,
        device_request=request,
        device_error=error,
    )


def test_local_copy_op_holds_slots_until_the_dma_lands() -> None:
    runtime = FakeRuntime(latency=2)
    gpu = _buffers(1, 9)
    dram = _buffers(1, 0)
    engine = _engine(runtime, *gpu)
    op = _copy_op(
        engine,
        [_descriptor(gpu[0], "VRAM")],
        [_descriptor(dram[0], "DRAM")],
        lambda: 0.0,
        10.0,
    )
    assert op.progress(None, None) == (False, True)
    assert op.close(None) is False, "closing must wait for the running copy"
    assert op.close(None) is True
    assert dram[0].raw == bytes([9]) * _SPAN
    assert op.transfer_id is None and op.device_handle is None


def test_local_copy_op_past_deadline_fails_after_the_dma_lands() -> None:
    runtime = FakeRuntime(latency=1)
    gpu = _buffers(1, 4)
    dram = _buffers(1, 0)
    engine = _engine(runtime, *gpu)
    now = [0.0]
    op = _copy_op(
        engine,
        [_descriptor(gpu[0], "VRAM")],
        [_descriptor(dram[0], "DRAM")],
        lambda: now[0],
        5.0,
    )
    assert op.progress(None, None) == (False, True)
    now[0] = 6.0
    assert op.progress(None, None) == (True, True)
    assert op.success is False and op.cancellation_requested
    assert dram[0].raw == bytes([4]) * _SPAN, "the copy still completed before release"


def test_local_copy_op_never_submits_after_the_deadline() -> None:
    runtime = FakeRuntime()
    gpu = _buffers(1, 4)
    dram = _buffers(1, 0)
    engine = _engine(runtime, *gpu)
    op = _copy_op(
        engine,
        [_descriptor(gpu[0], "VRAM")],
        [_descriptor(dram[0], "DRAM")],
        lambda: 9.0,
        5.0,
    )
    assert op.progress(None, None) == (True, True)
    assert op.success is False and runtime.calls == []


def test_local_copy_op_with_rejected_request_fails_without_cuda() -> None:
    runtime = FakeRuntime()
    gpu = _buffers(1, 4)
    stray = _buffers(1, 2)
    dram = _buffers(1, 0)
    engine = _engine(runtime, *gpu)
    op = _copy_op(
        engine,
        [_descriptor(stray[0], "VRAM")],
        [_descriptor(dram[0], "DRAM")],
        lambda: 0.0,
        5.0,
    )
    assert op.device_request is None and "not a registered" in (op.device_error or "")
    assert op.progress(None, None) == (True, True)
    assert op.success is False and runtime.calls == [] and runtime.devices == []
    assert op.close(None) is True


def test_engine_is_not_created_without_vram_regions() -> None:
    dram = _buffers(1, 0)
    region = FrameworkMemoryRegion(ctypes.addressof(dram[0]), _SPAN, "DRAM", 0)

    def refuse():
        raise AssertionError("runtime must not load without VRAM regions")

    with patch.object(CudaRuntime, "load", refuse):
        assert create_device_copy_engine((region,)) is None


def test_engine_falls_back_when_the_runtime_is_missing() -> None:
    gpu = _buffers(1, 0)

    def missing():
        raise OSError("libcudart not found")

    with patch.object(CudaRuntime, "load", missing):
        assert create_device_copy_engine((_region_over(gpu),)) is None


def test_engine_rejects_regions_on_invisible_devices() -> None:
    gpu = _buffers(1, 0)
    with patch.object(CudaRuntime, "load", lambda: FakeRuntime()):
        assert create_device_copy_engine((_region_over(gpu, device_id=5),)) is None
        engine = create_device_copy_engine((_region_over(gpu, device_id=1),))
    assert engine is not None and engine.devices == (1,)


def test_request_coalesces_spans_contiguous_on_both_sides() -> None:
    from kvcr.device_copy import _coalesce_operands

    # Three pages with adjacent device rows and adjacent slots merge into one
    # span; a gap on either side starts a new run; sizes add up exactly.
    size = 128
    dst = np.array([1000, 1128, 1256, 5000, 5128, 9000], dtype=np.uint64)
    src = np.array([2000, 2128, 2256, 6000, 6300, 9500], dtype=np.uint64)
    sizes = np.full(6, size, dtype=np.uint64)
    merged_dst, merged_src, merged_sizes = _coalesce_operands(dst, src, sizes)
    assert merged_dst.tolist() == [1000, 5000, 5128, 9000]
    assert merged_src.tolist() == [2000, 6000, 6300, 9500]
    assert merged_sizes.tolist() == [3 * size, size, size, size]
    assert int(merged_sizes.sum()) == int(sizes.sum())

    request = DeviceCopyEngine.request(0, dst, sizes, src, sizes)
    assert not isinstance(request, str)
    assert len(request.sizes) == 4 and request.byte_count == 6 * size
    # Fully contiguous operands collapse to one span; a single span is kept.
    one_dst, one_src, one_sizes = _coalesce_operands(dst[:3], src[:3], sizes[:3])
    assert one_dst.tolist() == [1000] and one_sizes.tolist() == [3 * size]
    assert _coalesce_operands(dst[:1], src[:1], sizes[:1])[2].tolist() == [size]


def test_small_copies_stay_on_one_stream() -> None:
    runtime = FakeRuntime(latency=1)
    count = _MIN_SPANS_PER_STREAM * _STREAMS_PER_DEVICE
    gpu = _buffers(count, 5)
    dram = _buffers(count, 0)
    engine = _engine(runtime, *gpu)
    # Many spans but few bytes: one batch call and one event, not four.
    handle = _submit(
        engine,
        [_descriptor(b, "VRAM") for b in gpu],
        [_descriptor(b, "DRAM") for b in dram],
    )
    assert len(handle.events) == 1
    assert len(runtime.calls) == 1
    assert _drain(engine, handle) is True
    assert all(buffer.raw == bytes([5]) * _SPAN for buffer in dram)
