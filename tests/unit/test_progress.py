# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from kvcr.progress import _KVCRProgress, _ProgressOp
from kvcr.types import MemDescriptor, RegDescriptor


class _TransferAgent:
    name = "transfer-test"

    def __init__(self) -> None:
        self.state = "DONE"
        self.submit_exception = False
        self.transfer_result = "PROC"
        self.release_failures = 0
        self.check_exception = False
        self.events: list[str] = []
        self.deregistered: list[int] = []
        self.prepared: set[str] = set()
        self.prep_calls: list[tuple[str, str, list[list[int]]]] = []
        self.make_calls: list[tuple[str, list[int], str, list[int]]] = []
        self.prep_failure: str | None = None
        self.dlist_release_failures = 0
        self._next = 0

    def prep_xfer_dlist(self, agent_name, descriptors, *, mem_type, backends):
        assert isinstance(descriptors, np.ndarray)
        assert descriptors.dtype == np.uint64
        assert descriptors.flags.c_contiguous and descriptors.shape[1] == 5
        if mem_type == self.prep_failure:
            raise RuntimeError("preparation failed")
        call = (agent_name, mem_type, descriptors.tolist())
        self.prep_calls.append(call)
        handle = f"dlist-{len(self.prep_calls)}"
        self.prepared.add(handle)
        return handle

    def make_prepped_xfer(
        self,
        operation,
        local_handle,
        local_indices,
        remote_handle,
        remote_indices,
        *,
        notif_msg,
        backends,
    ):
        self.make_calls.append(
            (local_handle, local_indices, remote_handle, remote_indices)
        )

        self._next += 1
        self.events.append(f"make:{operation}:{backends}:{notif_msg!r}")
        return self._next

    def release_dlist_handle(self, handle):
        self.events.append(f"release-dlist:{handle}")
        if self.dlist_release_failures:
            self.dlist_release_failures -= 1
            return False
        self.prepared.remove(handle)

    def get_xfer_descs(
        self, descriptors: list[tuple[int, int, int]], mem_type: str
    ) -> tuple[str, list[tuple[int, int, int]]]:
        self.events.append(f"describe:{mem_type}")
        return mem_type, descriptors

    def initialize_xfer(
        self,
        operation: str,
        local_descriptors: tuple[str, list[tuple[int, int, int]]],
        remote_descriptors: tuple[str, list[tuple[int, int, int]]],
        remote_side_agent: str | bytes,
        *,
        notif_msg: bytes,
        backends: list[str] | None = None,
    ) -> int:
        local = local_descriptors[1]
        remote = remote_descriptors[1]
        if len(local) != len(remote) or any(
            local_item[1] != remote_item[1]
            for local_item, remote_item in zip(local, remote)
        ):
            raise RuntimeError("NIXL rejected unaligned descriptors")
        self._next += 1
        self.events.append(
            "initialize:"
            f"{operation}:{local_descriptors[0]}:{remote_descriptors[0]}:"
            f"{remote_side_agent!r}:{backends}:{notif_msg!r}"
        )
        return self._next

    def transfer(self, handle: int) -> str:
        self.events.append(f"submit:{handle}")
        if self.submit_exception:
            raise RuntimeError("ambiguous submission")
        return self.transfer_result

    def check_xfer_state(self, handle: int) -> str:
        self.events.append(f"check:{handle}")
        if self.check_exception:
            raise RuntimeError("transfer failed")
        return self.state

    def get_xfer_telemetry(self, handle: int) -> SimpleNamespace:
        self.events.append(f"telemetry:{handle}")
        return SimpleNamespace(totalBytes=128)

    def release_xfer_handle(self, handle: int) -> bool:
        self.events.append(f"release-transfer:{handle}")
        if self.release_failures:
            self.release_failures -= 1
            return False
        return True

    def deregister_memory(self, handle: int) -> None:
        assert not self.prepared
        self.deregistered.append(handle)


def _mem(
    address: int,
    *,
    size: int = 128,
    mem_type: str = "DRAM",
    device_id: int = 0,
    info: str = "",
) -> MemDescriptor:
    return MemDescriptor("transfer-test", mem_type, address, size, device_id, info)


def _transfer_progress(agent: _TransferAgent, *, prepare: bool = True) -> _KVCRProgress:
    progress = _KVCRProgress(
        lambda _: None,
        lambda _, __: ({}, False),
        list,
        lambda: None,
        nixl_agent_name=agent.name,
    )
    progress._nixl_agent = agent
    if prepare:
        regions = (RegDescriptor(128, 128, count=4), RegDescriptor(256, 64))
        for name in ("", agent.name, "remote-agent"):
            progress.prepare_memory(name, regions)
    return progress


@pytest.mark.parametrize(
    ("transfer_result", "polls_state"),
    [("PROC", True), ("DONE", False)],
    ids=["async", "sync"],
)
def test_progress_submits_and_completes_transfer(
    transfer_result: str, polls_state: bool
) -> None:
    agent = _TransferAgent()
    agent.transfer_result = transfer_result
    progress = _transfer_progress(agent)
    transfer_id, submitted = progress.submit_transfer(
        "WRITE",
        (_mem(128),),
        (_mem(256),),
        remote_side_agent="remote-agent",
        notif_msg=b"done",
        capture_telemetry=True,
    )

    assert submitted
    assert "make:WRITE:[]:b'done'" in agent.events
    result = progress.poll_transfer(transfer_id)
    assert result is not None
    success, telemetry_result = result
    assert success
    assert any(event.startswith("check:") for event in agent.events) is polls_state
    assert telemetry_result is not None
    assert telemetry_result.totalBytes == 128
    telemetry = agent.events.index("telemetry:1")
    release = agent.events.index("release-transfer:1")
    assert telemetry < release


def test_progress_retries_release_while_operation_is_active() -> None:
    agent = _TransferAgent()
    agent.state = "PROC"
    agent.submit_exception = True
    agent.release_failures = 2
    progress = _transfer_progress(agent)
    transfer_id, submitted = progress.submit_transfer(
        "READ",
        (_mem(128),),
        (_mem(256),),
        remote_side_agent="remote-agent",
    )

    assert not submitted
    assert progress.poll_transfer(transfer_id, cancellation_requested=True) is None
    assert progress.poll_transfer(transfer_id, cancellation_requested=True) is None
    result = progress.poll_transfer(transfer_id, cancellation_requested=True)
    assert result is not None
    success, _ = result
    assert not success
    assert agent.events.count("release-transfer:1") == 3


def test_progress_reports_rejected_transfer_submission() -> None:
    agent = _TransferAgent()
    agent.transfer_result = "ERR"
    progress = _transfer_progress(agent)
    transfer_id, submitted = progress.submit_transfer(
        "READ",
        (_mem(128),),
        (_mem(256),),
        remote_side_agent="remote-agent",
    )

    assert not submitted
    result = progress.poll_transfer(transfer_id)
    assert result is not None
    success, _ = result
    assert not success


def test_progress_treats_poll_exception_as_terminal_failure() -> None:
    agent = _TransferAgent()
    agent.check_exception = True
    progress = _transfer_progress(agent)
    transfer_id, _ = progress.submit_transfer(
        "WRITE",
        (_mem(128),),
        (_mem(256),),
        remote_side_agent="remote-agent",
    )

    result = progress.poll_transfer(transfer_id)
    assert result is not None
    success, _ = result
    assert not success
    assert agent.events[-1] == "release-transfer:1"


def test_progress_cancel_retains_transfer_until_release_succeeds() -> None:
    agent = _TransferAgent()
    agent.release_failures = 1
    progress = _transfer_progress(agent)
    transfer_id, _ = progress.submit_transfer(
        "WRITE",
        (_mem(128),),
        (_mem(256),),
        remote_side_agent="remote-agent",
    )

    assert not progress.cancel_transfer(transfer_id)
    assert transfer_id in progress._active_transfers
    assert progress.cancel_transfer(transfer_id)
    assert transfer_id not in progress._active_transfers


def test_progress_supports_backend_scoped_local_g3_descriptors() -> None:
    agent = _TransferAgent()
    progress = _transfer_progress(agent)
    transfer_id, _ = progress.submit_transfer(
        "READ",
        (_mem(128),),
        (_mem(0, mem_type="FILE", device_id=7),),
        remote_side_agent=agent.name,
        backend="MOCK",
    )

    assert "initialize:READ:DRAM:FILE:'transfer-test':['MOCK']:b''" in agent.events
    result = progress.poll_transfer(transfer_id)
    assert result is not None
    success, _ = result
    assert success


@pytest.mark.parametrize(
    ("local", "remote", "remote_agent", "message"),
    [
        (
            (_mem(128), _mem(256, mem_type="VRAM")),
            (_mem(384), _mem(512)),
            "remote-agent",
            "cannot mix memory types",
        ),
        ((), (_mem(256),), "remote-agent", "non-empty"),
        ((_mem(128),), (_mem(256),), "", "remote-side agent"),
        ((_mem(129),), (_mem(256),), "remote-agent", "prepared"),
        ((_mem(640),), (_mem(256),), "remote-agent", "prepared"),
        ((_mem(128, size=64),), (_mem(256),), "remote-agent", "prepared"),
        ((_mem(128, device_id=1),), (_mem(256),), "remote-agent", "prepared"),
        ((_mem(128, mem_type="VRAM"),), (_mem(256),), "remote-agent", "prepared"),
    ],
)
def test_progress_rejects_invalid_transfer(
    local, remote, remote_agent, message
) -> None:
    agent = _TransferAgent()
    progress = _transfer_progress(agent)

    with pytest.raises(ValueError, match=message):
        progress.submit_transfer("WRITE", local, remote, remote_side_agent=remote_agent)

    assert not agent.make_calls
    assert not agent.events
    assert not progress._active_transfers


@pytest.mark.parametrize(
    ("local", "remote"),
    [
        ((_mem(128),), (_mem(256, size=64),)),
        ((_mem(128), _mem(256)), (_mem(384),)),
    ],
    ids=["size", "count"],
)
def test_progress_delegates_descriptor_alignment_to_nixl(local, remote) -> None:
    agent = _TransferAgent()
    agent.make_prepped_xfer = Mock(side_effect=RuntimeError("NIXL rejected unaligned"))
    progress = _transfer_progress(agent)

    with pytest.raises(RuntimeError, match="NIXL rejected unaligned"):
        progress.submit_transfer(
            "WRITE", local, remote, remote_side_agent="remote-agent"
        )

    agent.make_prepped_xfer.assert_called_once()
    assert progress._active_transfers == {}


def test_progress_reuses_strided_catalogs_with_physical_indices() -> None:
    agent = _TransferAgent()
    progress = _transfer_progress(agent, prepare=False)
    progress.prepare_memory(
        "",
        (
            RegDescriptor(1000, 16, info="pool:k", stride=64, count=3),
            RegDescriptor(2000, 16, info="pool:v", stride=32, count=2),
        ),
    )
    progress.prepare_memory(
        "remote-agent", (RegDescriptor(5000, 16, "VRAM", 3, stride=32, count=8),)
    )
    with pytest.raises(RuntimeError, match="already initialized"):
        progress.prepare_memory("remote-agent", ())
    local = (
        _mem(1064, size=16, info="another-label"),
        _mem(2032, size=16),
        _mem(1064, size=16, info="pool:k"),
    )
    remote = tuple(
        _mem(address, size=16, mem_type="VRAM", device_id=3)
        for address in (5096, 5000, 5096)
    )

    for _ in range(2):
        transfer_id, submitted = progress.submit_transfer(
            "WRITE", local, remote, remote_side_agent="remote-agent", backend="UCX"
        )
        assert submitted
        progress.poll_transfer(transfer_id)

    assert agent.prep_calls == [
        ("", "DRAM", [[1000, 16, 0, 64, 3], [2000, 16, 0, 32, 2]]),
        ("remote-agent", "VRAM", [[5000, 16, 3, 32, 8]]),
    ]
    assert agent.make_calls == [
        ("dlist-1", [1, 4, 1], "dlist-2", [3, 0, 3]),
        ("dlist-1", [1, 4, 1], "dlist-2", [3, 0, 3]),
    ]
    assert agent.events.count("make:WRITE:['UCX']:b''") == 2


@pytest.mark.parametrize("release_failures", [0, 2])
def test_progress_preparation_failure_is_fatal(release_failures) -> None:
    agent = _TransferAgent()
    progress = _transfer_progress(agent, prepare=False)
    progress._memory_regions = (
        RegDescriptor(128, 128),
        RegDescriptor(256, 128, "VRAM"),
    )
    agent.register_memory = Mock(side_effect=[7, 8])
    agent.prep_failure = "VRAM"
    agent.dlist_release_failures = release_failures

    with pytest.raises(RuntimeError, match="prepar"):
        progress.start()
    with pytest.raises(RuntimeError, match="prepar"):
        progress.close()

    assert bool(agent.prepared) == bool(release_failures)
    if release_failures:
        assert agent.deregistered == []
        assert progress.nixl_agent is agent
        assert not progress.is_quiescent()
        progress._close_nixl()
    assert not agent.prepared
    assert agent.deregistered == [8, 7]
    assert progress.is_quiescent()


def test_progress_close_retries_operations_until_they_release() -> None:
    class Operation(_ProgressOp):
        close_attempts = 0

        def progress(
            self,
            _progress: _KVCRProgress,
            _event: object | None,
        ) -> tuple[bool, bool]:
            return False, False

        def close(self, _progress: _KVCRProgress) -> bool:
            self.close_attempts += 1
            return self.close_attempts == 2

    progress = _transfer_progress(_TransferAgent())
    operation = Operation(op_id=("test", 1), keys=set())
    progress._in_flight_ops[operation.op_id] = operation

    progress._close_progress_ops()

    assert operation.close_attempts == 2
    assert progress._in_flight_ops == {}


@pytest.mark.parametrize(
    "attribute",
    ["_active_transfers", "_in_flight_ops", "_memory_registrations", "_prepared"],
)
def test_progress_quiescence_tracks_native_state(attribute: str) -> None:
    """Quiescence requires every native-state container to be empty."""
    progress = _transfer_progress(_TransferAgent(), prepare=False)
    held = getattr(progress, attribute)
    if isinstance(held, dict):
        held[0] = object()
    else:
        held.append(object())

    assert not progress.is_quiescent()
    held.clear()
    assert progress.is_quiescent()


def test_progress_cleanup_continues_after_operation_close_failure(
    monkeypatch,
) -> None:
    cleaned = []
    progress = _KVCRProgress(
        lambda _: None,
        lambda _, __: ({}, False),
        list,
        lambda: cleaned.append("backend"),
    )
    progress._stop_requested = True

    def fail_operation_cleanup() -> None:
        raise RuntimeError("operation cleanup failed")

    monkeypatch.setattr(progress, "_close_progress_ops", fail_operation_cleanup)
    monkeypatch.setattr(progress, "_close_nixl", lambda: cleaned.append("nixl"))

    progress._run()

    assert cleaned == ["backend", "nixl"]


def test_progress_does_not_deregister_memory_with_an_active_transfer() -> None:
    agent = _TransferAgent()
    progress = _transfer_progress(agent)
    progress._memory_registrations.append(7)
    transfer_id, _ = progress.submit_transfer(
        "WRITE",
        (_mem(128),),
        (_mem(256),),
        remote_side_agent="remote-agent",
    )
    agent.release_failures = 1
    assert progress.poll_transfer(transfer_id) is None

    with pytest.raises(RuntimeError, match="active transfers"):
        progress._close_nixl()
    with pytest.raises(RuntimeError, match="active transfers"):
        progress.release_prepared("remote-agent")

    assert progress.nixl_agent is agent
    assert agent.deregistered == []
    assert len(agent.prepared) == 3
    assert progress.cancel_transfer(transfer_id)

    progress._close_nixl()
    assert agent.deregistered == [7]
    assert progress._nixl_agent is None


@pytest.mark.parametrize(
    ("batch_size", "first_count"),
    [(2, 2), (0, 3)],
    ids=["bounded", "unlimited"],
)
def test_iteration_batching(batch_size: int, first_count: int) -> None:
    submitted = [object() for _ in range(3)]
    published = [object() for _ in range(3)]
    seen: list[object] = []
    outbound: list[object] = []

    def poll(
        _progress: _KVCRProgress,
        items: list[object],
    ) -> tuple[dict[object, object], bool]:
        if not seen:
            outbound.extend(published)
        seen.extend(items)
        return {}, bool(items)

    def flush() -> list[object]:
        items = list(outbound)
        outbound.clear()
        return items

    progress = _KVCRProgress(
        lambda _: None, poll, flush, lambda: None, batch_size=batch_size
    )
    for item in submitted:
        progress.submit(item)

    assert progress._run_one_iteration()
    assert seen == submitted[:first_count]
    assert progress.take_completed() == published[:first_count]

    assert progress._run_one_iteration() is (first_count < len(submitted))
    assert seen == submitted
    assert progress.take_completed() == published[first_count:]
    assert progress.take_completed() == []


def test_real_thread_owns_lifecycle_and_transfers_same_object() -> None:
    main_thread = threading.get_ident()
    lifecycle_threads: list[int] = []
    stepped = threading.Event()
    update = object()
    outbound: list[object] = []

    def initialize(_progress: _KVCRProgress) -> None:
        lifecycle_threads.append(threading.get_ident())

    class Operation(_ProgressOp):
        def progress(
            self,
            _progress: _KVCRProgress,
            event: object | None,
        ) -> tuple[bool, bool]:
            assert event is None
            lifecycle_threads.append(threading.get_ident())
            outbound.append(update)
            stepped.set()
            return True, True

        def close(self, _progress: _KVCRProgress) -> bool:
            lifecycle_threads.append(threading.get_ident())
            return True

    item = Operation(op_id=("test", 1), keys=set())
    assert isinstance(item, _ProgressOp)

    def poll(
        _progress: _KVCRProgress,
        items: list[object],
    ) -> tuple[dict[object, object], bool]:
        assert items == []
        return {}, False

    def flush() -> list[object]:
        items = list(outbound)
        outbound.clear()
        return items

    def close() -> None:
        lifecycle_threads.append(threading.get_ident())

    progress = _KVCRProgress(initialize, poll, flush, close)
    progress.start()
    progress.submit(item)
    assert stepped.wait(timeout=1)
    progress.close()

    completed = progress.take_completed()
    assert completed == [update, item]
    assert completed[1] is item
    assert lifecycle_threads
    assert len(set(lifecycle_threads)) == 1
    assert lifecycle_threads[0] != main_thread


def test_startup_failure_is_reported_to_main() -> None:
    expected = RuntimeError("startup failed")

    def initialize(_progress: _KVCRProgress) -> None:
        raise expected

    progress = _KVCRProgress(
        initialize,
        lambda _, items: ({}, False),
        lambda: [],
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="startup failed") as exc_info:
        progress.start()
    assert exc_info.value is expected


def test_loop_failure_is_reported_to_main() -> None:
    expected = RuntimeError("loop failed")
    stepped = threading.Event()

    def poll(
        _progress: _KVCRProgress,
        items: list[object],
    ) -> tuple[dict[object, object], bool]:
        if items:
            stepped.set()
            raise expected
        return {}, False

    progress = _KVCRProgress(lambda _: None, poll, lambda: [], lambda: None)
    progress.start()
    progress.submit(object())
    assert stepped.wait(timeout=1)

    with pytest.raises(RuntimeError, match="loop failed") as exc_info:
        progress.close()
    assert exc_info.value is expected
