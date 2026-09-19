"""Same-process device copies for local DRAM fills and deliveries.

NIXL routes a transfer whose two endpoints belong to the same agent through
its network transport. With UCX that loopback has no native lane for device
memory: a put between VRAM and host memory falls back to software emulation
over TCP and runs at a small fraction of PCIe bandwidth. A copy that never
leaves the process only needs the CUDA runtime, so local DRAM copies with a
VRAM endpoint are issued as ``cudaMemcpyBatchAsync`` calls on KVCR-owned
streams and complete through recorded events.

Span lists run to thousands of entries per operation (one span per layer
buffer per page), so validation and argument marshalling are vectorised with
numpy rather than iterated in Python.

The engine binds the CUDA runtime through ctypes so KVCR stays free of any
framework dependency. It is only created when VRAM framework regions exist and
is driven exclusively by the progress thread.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path

import numpy as np

from .config import FrameworkMemoryRegion
from .types import MemDescriptor

logger = logging.getLogger(__name__)

_SUCCESS = 0
_ERROR_INVALID_VALUE = 1
_ERROR_NOT_READY = 600
_ERROR_NOT_SUPPORTED = 801
_MEMCPY_DEFAULT = 4
_STREAM_NON_BLOCKING = 0x01
_EVENT_DISABLE_TIMING = 0x02
_SRC_ACCESS_ORDER_STREAM = 0x1

# Copy engines are underused by one queue of small spans; a handful of streams
# recovers most of the gap to a single large memcpy. Splitting a small batch
# only adds launches, so batches below the threshold stay on one stream.
_STREAMS_PER_DEVICE = 4
_MIN_SPANS_PER_STREAM = 64

_LIBRARY_NAMES = ("libcudart.so.13", "libcudart.so.12", "libcudart.so")
# Python wheels carry the runtime under site-packages when no system CUDA
# toolkit is installed.
_WHEEL_LIBRARY_PATHS = (
    "nvidia/cu13/lib/libcudart.so.13",
    "nvidia/cuda_runtime/lib/libcudart.so.12",
)

_addr = attrgetter("addr")
_size = attrgetter("size")
_mem_type = attrgetter("mem_type")
_device = attrgetter("device_Id")

_PointerArray = ctypes.POINTER(ctypes.c_void_p)
_SizeArray = ctypes.POINTER(ctypes.c_size_t)


class CudaError(RuntimeError):
    def __init__(self, what: str, code: int, message: str) -> None:
        super().__init__(f"{what} failed: {message} ({code})")
        self.code = code


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _MemLocation),
        ("dstLocHint", _MemLocation),
        ("flags", ctypes.c_uint),
    ]


class CudaRuntime:
    """The CUDA runtime entry points the copy engine uses."""

    def __init__(
        self, library: ctypes.CDLL, releasing_library: ctypes.CDLL | None = None
    ) -> None:
        # ``library`` keeps the GIL across each call (PyDLL); the batch copy,
        # whose driver-side validation grows with the span count, is bound
        # through ``releasing_library`` (CDLL) so a long enqueue does not
        # stall other Python threads.
        self._lib = library
        self._releasing_lib = releasing_library or library
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        self._bind("cudaSetDevice", [ctypes.c_int])
        self._bind("cudaGetDeviceCount", [ctypes.POINTER(ctypes.c_int)])
        self._bind(
            "cudaStreamCreateWithFlags", [ctypes.POINTER(pointer), ctypes.c_uint]
        )
        self._bind("cudaStreamDestroy", [pointer])
        self._bind("cudaEventCreateWithFlags", [ctypes.POINTER(pointer), ctypes.c_uint])
        self._bind("cudaEventDestroy", [pointer])
        self._bind("cudaEventRecord", [pointer, pointer])
        self._bind("cudaEventQuery", [pointer])
        self._bind("cudaMemcpyAsync", [pointer, pointer, size, ctypes.c_int, pointer])
        library.cudaGetErrorString.argtypes = [ctypes.c_int]
        library.cudaGetErrorString.restype = ctypes.c_char_p
        self.has_memcpy_batch = hasattr(library, "cudaMemcpyBatchAsync")
        if self.has_memcpy_batch:
            self._bind(
                "cudaMemcpyBatchAsync",
                [
                    _PointerArray,
                    _PointerArray,
                    _SizeArray,
                    size,
                    ctypes.POINTER(_MemcpyAttributes),
                    _SizeArray,
                    size,
                    pointer,
                ],
                library=self._releasing_lib,
            )
        self._batch_attributes = (_MemcpyAttributes * 1)()
        self._batch_attributes[0].srcAccessOrder = _SRC_ACCESS_ORDER_STREAM
        self._batch_attribute_index = (ctypes.c_size_t * 1)(0)

    def _bind(
        self, name: str, argtypes: list, library: ctypes.CDLL | None = None
    ) -> None:
        function = getattr(library if library is not None else self._lib, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int

    @classmethod
    def load(cls) -> CudaRuntime:
        candidates = list(_LIBRARY_NAMES)
        for root in sys.path:
            for relative in _WHEEL_LIBRARY_PATHS:
                path = Path(root) / relative
                if path.is_file():
                    candidates.append(str(path))
        errors: list[str] = []
        for candidate in candidates:
            try:
                # PyDLL keeps the GIL across the microsecond calls: releasing
                # it for every one of them would cost a switch interval to get
                # it back whenever another Python thread is busy, which is the
                # normal state of a serving process. The batch copy alone goes
                # through a releasing handle.
                return cls(ctypes.PyDLL(candidate), ctypes.CDLL(candidate))
            except (OSError, AttributeError) as error:
                errors.append(f"{candidate}: {error}")
        raise OSError("CUDA runtime library not found: " + "; ".join(errors))

    def error_string(self, code: int) -> str:
        raw = self._lib.cudaGetErrorString(code)
        return raw.decode() if raw else f"CUDA error {code}"

    def check(self, code: int, what: str) -> None:
        if code != _SUCCESS:
            raise CudaError(what, code, self.error_string(code))

    def device_count(self) -> int:
        count = ctypes.c_int(0)
        self.check(
            self._lib.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount"
        )
        return count.value

    def set_device(self, device_id: int) -> None:
        self.check(self._lib.cudaSetDevice(device_id), "cudaSetDevice")

    def stream_create(self) -> int:
        stream = ctypes.c_void_p()
        self.check(
            self._lib.cudaStreamCreateWithFlags(
                ctypes.byref(stream), _STREAM_NON_BLOCKING
            ),
            "cudaStreamCreateWithFlags",
        )
        return stream.value or 0

    def stream_destroy(self, stream: int) -> int:
        return self._lib.cudaStreamDestroy(stream)

    def event_create(self) -> int:
        event = ctypes.c_void_p()
        self.check(
            self._lib.cudaEventCreateWithFlags(
                ctypes.byref(event), _EVENT_DISABLE_TIMING
            ),
            "cudaEventCreateWithFlags",
        )
        return event.value or 0

    def event_destroy(self, event: int) -> int:
        return self._lib.cudaEventDestroy(event)

    def event_record(self, event: int, stream: int) -> None:
        self.check(self._lib.cudaEventRecord(event, stream), "cudaEventRecord")

    def event_query(self, event: int) -> int:
        return self._lib.cudaEventQuery(event)

    def memcpy_async(self, dst: int, src: int, size: int, stream: int) -> int:
        return self._lib.cudaMemcpyAsync(dst, src, size, _MEMCPY_DEFAULT, stream)

    def memcpy_batch_async(
        self, dsts: np.ndarray, srcs: np.ndarray, sizes: np.ndarray, stream: int
    ) -> int:
        """Issue one batch from contiguous uint64 address and size arrays."""
        return self._releasing_lib.cudaMemcpyBatchAsync(
            dsts.ctypes.data_as(_PointerArray),
            srcs.ctypes.data_as(_PointerArray),
            sizes.ctypes.data_as(_SizeArray),
            len(dsts),
            self._batch_attributes,
            self._batch_attribute_index,
            1,
            stream,
        )


@dataclass
class DeviceCopyHandle:
    """One submitted operation: completes when every recorded event is done."""

    device_id: int
    byte_count: int
    events: list[int] = field(default_factory=list)
    error: str | None = None


def _coalesce_operands(
    dst: np.ndarray, src: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge runs of consecutive spans contiguous in both address spaces.

    Vectorised: a break starts wherever the next span does not begin exactly
    where the previous one ends on either side; runs are then reduced with
    one cumulative sum, so the cost stays a few microseconds per thousand
    spans while the batch copy's per-span host cost drops with the count.
    """
    count = len(sizes)
    if count < 2:
        return dst, src, sizes
    dst = np.asarray(dst, dtype=np.uint64)
    src = np.asarray(src, dtype=np.uint64)
    sizes = np.asarray(sizes, dtype=np.uint64)
    ends_dst = dst[:-1] + sizes[:-1]
    ends_src = src[:-1] + sizes[:-1]
    breaks = (dst[1:] != ends_dst) | (src[1:] != ends_src)
    if not breaks.any():
        return dst[:1], src[:1], np.array([sizes.sum()], dtype=np.uint64)
    starts = np.concatenate(([0], np.flatnonzero(breaks) + 1))
    cumulative = np.cumsum(sizes, dtype=np.uint64)
    run_ends = cumulative[np.concatenate((starts[1:] - 1, [count - 1]))]
    run_begins = cumulative[starts] - sizes[starts]
    return dst[starts], src[starts], (run_ends - run_begins).astype(np.uint64)


@dataclass(frozen=True)
class DeviceCopyRequest:
    """Aligned copy operands for one operation, validated before submission."""

    device_id: int
    dst_addresses: np.ndarray
    src_addresses: np.ndarray
    sizes: np.ndarray

    @property
    def byte_count(self) -> int:
        return int(self.sizes.sum())


class DeviceCopyEngine:
    """Issue and track CUDA copies between registered VRAM regions and local DRAM."""

    def __init__(
        self, runtime: CudaRuntime, regions: Iterable[FrameworkMemoryRegion]
    ) -> None:
        self._runtime = runtime
        spans: dict[int, list[tuple[int, int]]] = {}
        for region in regions:
            if region.mem_type == "VRAM":
                spans.setdefault(region.device_id, []).append(
                    (region.address, region.end)
                )
        # Registered regions never partially overlap, so sorted starts index
        # the region an address falls in with one search.
        self._region_starts: dict[int, np.ndarray] = {}
        self._region_ends: dict[int, np.ndarray] = {}
        for device, items in spans.items():
            items.sort()
            self._region_starts[device] = np.fromiter(
                (start for start, _ in items), dtype=np.uint64, count=len(items)
            )
            self._region_ends[device] = np.fromiter(
                (end for _, end in items), dtype=np.uint64, count=len(items)
            )
        self._streams: dict[int, list[int]] = {}
        self._free_events: dict[int, list[int]] = {}
        self._use_batch = runtime.has_memcpy_batch
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.bytes_completed = 0

    @property
    def devices(self) -> tuple[int, ...]:
        return tuple(sorted(self._region_starts))

    @property
    def uses_batch_memcpy(self) -> bool:
        return self._use_batch

    def covers(self, descriptor: MemDescriptor) -> bool:
        """Whether a VRAM span lies inside one registered framework region."""
        addresses = np.array([descriptor.addr], dtype=np.uint64)
        sizes = np.array([descriptor.size], dtype=np.uint64)
        return self._unregistered(descriptor.device_Id, addresses, sizes) is None

    def span_arrays(
        self, spans: Sequence[MemDescriptor]
    ) -> tuple[np.ndarray, np.ndarray] | str:
        """Addresses and sizes of framework spans, or why they cannot be copied.

        VRAM spans must lie inside registered framework regions; the check is
        vectorised because a page contributes one span per layer buffer.
        """
        count = len(spans)
        if not count:
            return "a copy needs at least one span"
        if len(set(map(_mem_type, spans))) != 1 or len(set(map(_device, spans))) != 1:
            return "one copy cannot mix memory types or devices"
        addresses = np.fromiter(map(_addr, spans), dtype=np.uint64, count=count)
        sizes = np.fromiter(map(_size, spans), dtype=np.uint64, count=count)
        if spans[0].mem_type == "VRAM":
            problem = self._unregistered(spans[0].device_Id, addresses, sizes)
            if problem is not None:
                return problem
        return addresses, sizes

    @staticmethod
    def request(
        device_id: int,
        dst_addresses: np.ndarray,
        dst_sizes: np.ndarray,
        src_addresses: np.ndarray,
        src_sizes: np.ndarray,
    ) -> DeviceCopyRequest | str:
        """Pair aligned operand arrays, or say why they do not align.

        Consecutive spans that are contiguous on both sides are merged: the
        CUDA batch copy costs host time per span, and pages allocated in
        sequence usually have adjacent device rows and adjacent slots.
        """
        if len(src_sizes) != len(dst_sizes):
            return "source and destination span counts differ"
        if not np.array_equal(src_sizes, dst_sizes):
            index = int(np.argmax(src_sizes != dst_sizes))
            return f"span sizes differ ({src_sizes[index]} vs {dst_sizes[index]})"
        dst_addresses, src_addresses, sizes = _coalesce_operands(
            dst_addresses, src_addresses, src_sizes
        )
        return DeviceCopyRequest(device_id, dst_addresses, src_addresses, sizes)

    def submit(self, request: DeviceCopyRequest) -> DeviceCopyHandle:
        """Enqueue a request on device streams and record completion events."""
        device_id = request.device_id
        handle = DeviceCopyHandle(device_id, request.byte_count)
        self.submitted += 1
        runtime = self._runtime
        streams: list[int] = []
        try:
            runtime.set_device(device_id)
            streams = self._streams_for(device_id, len(request.sizes))
            if len(streams) == 1:
                self._enqueue(
                    request.dst_addresses,
                    request.src_addresses,
                    request.sizes,
                    streams[0],
                )
            else:
                for lane, (dsts, srcs, lengths) in enumerate(
                    zip(
                        np.array_split(request.dst_addresses, len(streams)),
                        np.array_split(request.src_addresses, len(streams)),
                        np.array_split(request.sizes, len(streams)),
                    )
                ):
                    self._enqueue(dsts, srcs, lengths, streams[lane])
        except CudaError as error:
            handle.error = str(error)
        # Spans accepted before a failure are still in flight, so events are
        # recorded regardless and the slots stay held until they fire.
        for stream in streams:
            try:
                event = self._event(device_id)
                runtime.event_record(event, stream)
                handle.events.append(event)
            except CudaError as error:
                handle.error = handle.error or str(error)
        return handle

    def poll(self, handle: DeviceCopyHandle) -> bool | None:
        """None while any copy runs, else whether every span landed."""
        while handle.events:
            code = self._runtime.event_query(handle.events[-1])
            if code == _ERROR_NOT_READY:
                return None
            event = handle.events.pop()
            self._free_events.setdefault(handle.device_id, []).append(event)
            if code != _SUCCESS:
                handle.error = handle.error or self._runtime.error_string(code)
        if handle.error is None:
            self.completed += 1
            self.bytes_completed += handle.byte_count
            return True
        self.failed += 1
        logger.warning("KVCR device copy failed: %s", handle.error)
        return False

    def close(self) -> None:
        runtime = self._runtime
        for device_id, streams in self._streams.items():
            try:
                runtime.set_device(device_id)
            except CudaError:
                logger.debug("KVCR device copy close skipped device %d", device_id)
                continue
            for event in self._free_events.pop(device_id, []):
                runtime.event_destroy(event)
            for stream in streams:
                runtime.stream_destroy(stream)
        self._streams.clear()
        self._free_events.clear()

    def _unregistered(
        self, device_id: int, addresses: np.ndarray, sizes: np.ndarray
    ) -> str | None:
        starts = self._region_starts.get(device_id)
        if starts is None:
            return f"no VRAM framework region is registered on device {device_id}"
        ends = self._region_ends[device_id]
        index = np.searchsorted(starts, addresses, side="right") - 1
        inside = index >= 0
        index = np.clip(index, 0, len(starts) - 1)
        inside &= addresses + sizes <= ends[index]
        if inside.all():
            return None
        bad = int(np.argmin(inside))
        return (
            f"VRAM span {int(addresses[bad]):#x}+{int(sizes[bad])} on device "
            f"{device_id} is not a registered framework region"
        )

    def _streams_for(self, device_id: int, span_count: int) -> list[int]:
        wanted = max(1, min(_STREAMS_PER_DEVICE, span_count // _MIN_SPANS_PER_STREAM))
        streams = self._streams.setdefault(device_id, [])
        while len(streams) < wanted:
            streams.append(self._runtime.stream_create())
        return streams[:wanted]

    def _event(self, device_id: int) -> int:
        free = self._free_events.get(device_id)
        if free:
            return free.pop()
        return self._runtime.event_create()

    def _enqueue(
        self, dsts: np.ndarray, srcs: np.ndarray, sizes: np.ndarray, stream: int
    ) -> None:
        runtime = self._runtime
        if len(sizes) > 1 and self._use_batch:
            code = runtime.memcpy_batch_async(dsts, srcs, sizes, stream)
            if code == _SUCCESS:
                return
            if code not in (_ERROR_INVALID_VALUE, _ERROR_NOT_SUPPORTED):
                runtime.check(code, "cudaMemcpyBatchAsync")
            # A rejected batch enqueues nothing, so per-span copies can take over.
            self._use_batch = False
            logger.warning(
                "cudaMemcpyBatchAsync unavailable (%s); using per-span cudaMemcpyAsync",
                runtime.error_string(code),
            )
        for dst, src, size in zip(dsts.tolist(), srcs.tolist(), sizes.tolist()):
            runtime.check(
                runtime.memcpy_async(dst, src, size, stream), "cudaMemcpyAsync"
            )


def create_device_copy_engine(
    regions: Iterable[FrameworkMemoryRegion],
) -> DeviceCopyEngine | None:
    """Build the engine when VRAM regions exist and the CUDA runtime loads."""
    regions = tuple(regions)
    if not any(region.mem_type == "VRAM" for region in regions):
        return None
    try:
        runtime = CudaRuntime.load()
        device_count = runtime.device_count()
    except (OSError, CudaError) as error:
        logger.warning(
            "KVCR device copy engine unavailable (%s); VRAM copies use NIXL loopback",
            error,
        )
        return None
    engine = DeviceCopyEngine(runtime, regions)
    unknown = [device for device in engine.devices if device >= device_count]
    if unknown:
        logger.warning(
            "KVCR device copy engine unavailable: VRAM regions name devices %s but "
            "only %d CUDA devices are visible; VRAM copies use NIXL loopback",
            unknown,
            device_count,
        )
        return None
    logger.info(
        "KVCR device copy engine active for CUDA devices %s (batch memcpy: %s)",
        list(engine.devices),
        runtime.has_memcpy_batch,
    )
    return engine
