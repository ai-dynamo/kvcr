# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared fakes and builders for KVCR integration-style unit tests."""

import ctypes
import random
import socket
import time
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import msgspec

from kvcr import KVCR, KVCRBindings
from kvcr import progress as kvcr_progress
from kvcr.config import (
    FrameworkDramInput,
    G3Options,
    KVCRBackendConfigs,
    KVCRConfig,
    LocalDramInfo,
    RemoteFWDramOptions,
)
from kvcr.core import _BlockRecord
from kvcr.kv_hints import KvSourceLocationsHint
from kvcr.local_disk import _G3Residency
from kvcr.local_dram import _LocalDramResidency, _LocalDramState
from kvcr.policy import FIFOPolicy
from kvcr.types import (
    BlockKey,
    MemDescriptor,
    OpEntryResult,
    OpEntryStatus,
    PinHandle,
    PinRequestId,
)

_OPEN_KVCRS: list[KVCR] = []


_HANDED_OUT: set[int] = set()


def _ephemeral_floor() -> int:
    """The lowest port the kernel hands out for a bind to port 0."""
    try:
        text = Path("/proc/sys/net/ipv4/ip_local_port_range").read_text()
        return int(text.split()[0])
    except (OSError, ValueError, IndexError):
        return 32768


@contextmanager
def listening_socket(port: int = 0) -> Iterator[socket.socket]:
    with socket.create_server(("127.0.0.1", port)) as listener:
        yield listener


def free_port() -> int:
    """A port this suite can bind now and still find free in a moment."""
    ceiling = _ephemeral_floor() - 1
    floor = max(1024, ceiling // 2)
    for _ in range(50):
        port = random.randint(floor, ceiling)
        # Picking a port can repeat; a test that asks twice must not get one twice.
        if port in _HANDED_OUT:
            continue
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        _HANDED_OUT.add(port)
        return port
    raise RuntimeError(f"no free port in [{floor}, {ceiling}]")


class FakeTelemetryStats:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, int | float, tuple[str, ...]]] = []

    def increase_counter(
        self,
        name: str,
        value: int | float = 1,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        self.records.append(("counter", name, value, labelvalues))

    def set_gauge(
        self,
        name: str,
        value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        self.records.append(("gauge", name, value, labelvalues))

    def observe_histogram(
        self,
        name: str,
        value: int | float,
        labelvalues: tuple[str, ...] = (),
    ) -> None:
        self.records.append(("histogram", name, value, labelvalues))

    def reduce(self) -> dict[str, int | float]:
        return {}

    def is_empty(self) -> bool:
        return not self.records


@contextmanager
def _use_nixl_agent(agent):
    def create_agent(name, *_):
        agent.name = name
        return agent

    with patch.multiple(
        kvcr_progress,
        nixl_agent=create_agent,
        nixl_agent_config=lambda **kwargs: kwargs,
    ):
        yield


def _poll_until(
    kvcr: KVCR,
    predicate,
    *,
    timeout: float = 1,
):
    completed = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        completed.extend(kvcr.poll_completed())
        if predicate(completed):
            return completed
        time.sleep(0.001)
    raise AssertionError("KVCR progress condition was not reached")


def _has_outstanding_operations(kvcr: KVCR) -> bool:
    return kvcr._core._outstanding_operations > 0


def _wait_until(predicate, *, timeout: float = 1) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("KVCR progress condition was not reached")


def _block_op_ids(kvcr: KVCR) -> set[tuple[str, Any]]:
    return {
        op_id
        for record in kvcr._core._block_record_map.values()
        for op_id in record.active_op_ids
    }


class FakePrimaryPinning:
    def __init__(
        self,
        prefix_length: int | None = None,
        missing_indices: Collection[int] = (),
    ):
        self.searches: list[tuple[BlockKey, ...]] = []
        self.unpins: list[PinHandle] = []
        self._next_request_id = 0
        self.completed = []
        self.prefix_length = prefix_length
        self.missing_indices = set(missing_indices)

    def request_pin(self, keys: Collection[BlockKey]) -> PinRequestId:
        keys = tuple(keys)
        self.searches.append(keys)
        request = PinRequestId(self._next_request_id)
        self._next_request_id += 1
        prefix_length = self.prefix_length
        if prefix_length == 0:
            self.completed.append((request, None))
            return request
        self.completed.append(
            (
                request,
                (
                    "pin",
                    {
                        key: (
                            _mem_descriptor(addr=0)
                            if (
                                (prefix_length is None or index < prefix_length)
                                and index not in self.missing_indices
                            )
                            else None
                        )
                        for index, key in enumerate(keys)
                    },
                ),
            )
        )
        return request

    def poll_pin_results(self):
        completed = self.completed
        self.completed = []
        return completed

    def release_pin(self, pin_handle: PinHandle) -> bool:
        self.unpins.append(pin_handle)
        return pin_handle == "pin"


class PendingPrimaryPinning(FakePrimaryPinning):
    def __init__(self):
        super().__init__()
        self.pending: dict[PinRequestId, tuple[BlockKey, ...]] = {}
        self.cancelled: list[PinRequestId] = []

    def request_pin(self, keys: Collection[BlockKey]) -> PinRequestId:
        keys = tuple(keys)
        self.searches.append(keys)
        request = PinRequestId(self._next_request_id)
        self._next_request_id += 1
        self.pending[request] = keys
        return request

    def complete(
        self,
        request_id: int,
        pin_handle: PinHandle = "pin",
        missing_indices: Collection[int] = (),
    ) -> None:
        request = PinRequestId(request_id)
        keys = self.pending.pop(request)
        missing_indices = set(missing_indices)
        self.completed.append(
            (
                request,
                (
                    pin_handle,
                    {
                        key: None if index in missing_indices else _mem_descriptor()
                        for index, key in enumerate(keys)
                    },
                ),
            )
        )

    def release_pin(self, pin_handle: PinHandle) -> bool:
        self.unpins.append(pin_handle)
        return pin_handle == "pin" or pin_handle.startswith("pin-")

    def cancel_pin_request(self, request: PinRequestId) -> None:
        self.cancelled.append(request)


class FakeNixlAgent:
    def __init__(self, metadata: bytes = b"metadata"):
        self.name = ""
        self.metadata = metadata
        self.registrations: list[tuple[Any, str]] = []
        self.deregistered: list[Any] = []
        self.remote_agents: list[bytes] = []
        self.xfers: list[tuple[Any, ...]] = []
        self.xfer_backends: list[list[str] | None] = []
        self.transfers: list[int] = []
        self.sent_notifs: list[tuple[bytes, bytes]] = []
        self.released_xfers: list[int] = []
        self.notifs: dict[str, list[bytes]] = {}
        self.telemetry_handles: list[int] = []
        self.state = "PROC"

    def register_memory(self, descs, mem_type="DRAM"):
        self.registrations.append((list(descs), mem_type))
        return len(self.registrations)

    def deregister_memory(self, handle):
        self.deregistered.append(handle)

    def get_agent_metadata(self) -> bytes:
        return self.metadata

    def add_remote_agent(self, metadata: bytes) -> bytes:
        self.remote_agents.append(metadata)
        return f"remote-{len(self.remote_agents)}".encode()

    def get_xfer_descs(self, descs, mem_type="DRAM"):
        return list(descs)

    def initialize_xfer(
        self,
        op,
        local_descs,
        remote_descs,
        remote_agent,
        notif_msg=b"",
        backends=None,
    ):
        local_descs = list(local_descs)
        remote_descs = list(remote_descs)
        self.xfers.append(
            (
                op,
                local_descs,
                list(range(len(local_descs))),
                remote_descs,
                remote_agent,
                notif_msg,
            )
        )
        self.xfer_backends.append(backends)
        return len(self.xfers)

    def transfer(self, handle):
        self.transfers.append(handle)
        op, local_descs, local_indices, remote_descs, remote_agent, _ = self.xfers[
            handle - 1
        ]
        if op == "WRITE" and remote_agent == self.name:
            for local_index, remote_index in zip(local_indices, local_indices):
                src_addr, src_size, _ = local_descs[local_index]
                dst_addr, dst_size, _ = remote_descs[remote_index]
                ctypes.memmove(dst_addr, src_addr, min(src_size, dst_size))
        return "PROC"

    def check_xfer_state(self, handle):
        return self.state

    def get_xfer_telemetry(self, handle):
        self.telemetry_handles.append(handle)
        return SimpleNamespace(
            xferDuration=2000,
            postDuration=500,
            totalBytes=32,
        )

    def release_xfer_handle(self, handle):
        self.released_xfers.append(handle)

    def send_notif(self, agent_name, notif_msg):
        self.sent_notifs.append((agent_name, notif_msg))

    def get_new_notifs(self):
        notifs = self.notifs
        self.notifs = {}
        return notifs


class FakeBytesControl:
    def __init__(self, endpoint: str = "tcp://target:1"):
        self.endpoint = endpoint
        self.sent: list[tuple[str, bytes]] = []
        self.incoming: list[bytes] = []
        self.send_result = True

    def send(self, endpoint: str, message: bytes) -> bool:
        self.sent.append((endpoint, message))
        return self.send_result

    def recv(self) -> list[bytes]:
        incoming = self.incoming
        self.incoming = []
        return incoming


def _mem_descriptor(addr: int = 128, size: int = 16) -> MemDescriptor:
    return MemDescriptor(
        end_point_name="primary",
        mem_type="DRAM",
        addr=addr,
        size=size,
        device_Id=0,
        info="",
    )


def _decode_control_message(message: bytes) -> dict[str, Any]:
    return msgspec.msgpack.decode(message)


def _decode_notif(notif: bytes) -> dict[str, Any]:
    assert notif.startswith(b"KVCR:")
    return msgspec.msgpack.decode(notif[len(b"KVCR:") :])


def _write_done_notification(
    op_handle: int,
    *,
    success: bool = True,
    completed_count: int | None = None,
) -> bytes:
    payload = {
        "type": "write_done",
        "op_handle": op_handle,
        "success": success,
    }
    if completed_count is not None:
        payload["completed_count"] = completed_count
    return b"KVCR:" + msgspec.msgpack.encode(payload)


def _op_entries(
    entries: Mapping[BlockKey, bool],
) -> dict[BlockKey, OpEntryResult]:
    return {
        key: OpEntryResult(OpEntryStatus.SUCCESS if success else OpEntryStatus.FAILED)
        for key, success in entries.items()
    }


def _start_write_message(
    op_handle: int,
    key: bytes,
    *,
    target_agent: str | None = None,
    remaining_timeout_ms: float = 1000,
) -> bytes:
    payload: dict[str, Any] = {
        "type": "start_write",
        "op_handle": op_handle,
        "remaining_timeout_ms": remaining_timeout_ms,
        "target_agent_metadata": b"target-md",
        "keys": [key],
        "dst_descriptors": [_mem_descriptor().__dict__],
    }
    if target_agent is not None:
        payload["target_agent"] = target_agent
    return msgspec.msgpack.encode(payload)


def _new_kvcr(
    agent: FakeNixlAgent,
    pinning: FakePrimaryPinning,
    control: FakeBytesControl,
    config: KVCRConfig | None = None,
    name: str = "target",
    key_hint_adapter: object | None = None,
    remote_options: RemoteFWDramOptions | None = None,
    framework_dram: FrameworkDramInput | None = None,
    local_dram: LocalDramInfo | None = None,
    g3: G3Options | None = None,
    inventory_sink=None,
    policy=None,
) -> KVCR:
    config = replace(
        config or KVCRConfig(nixl_agent_name=name, inventory_report_interval_ms=0),
        nixl_agent_name=name,
        nixl_listen_port=1,
    )
    with _use_nixl_agent(agent):
        kvcr = KVCR(
            config,
            KVCRBindings(
                request_pin=pinning.request_pin,
                poll_pin_results=pinning.poll_pin_results,
                release_pin=pinning.release_pin,
                cancel_pin_request=getattr(pinning, "cancel_pin_request", None),
                framework_control=control,
                key_hint_adapter=key_hint_adapter,
                inventory_sink=inventory_sink,
                policy=policy,
                stats_factory=(FakeTelemetryStats if config.enable_telemetry else None),
            ),
            KVCRBackendConfigs(
                framework_dram=framework_dram,
                local_dram=local_dram,
                g3=g3,
                remote_fw_dram=remote_options or RemoteFWDramOptions(),
            ),
        )
    _OPEN_KVCRS.append(kvcr)
    return kvcr


def _new_local_kvcr(
    agent,
    local,
    slot_count,
    inventory_sink=None,
    capacity_low_watermark_percent=0,
    capacity_needed_callback=None,
    policy=None,
) -> KVCR:
    pinning = FakePrimaryPinning()
    with _use_nixl_agent(agent):
        kvcr = KVCR(
            KVCRConfig(
                nixl_agent_name="target",
                nixl_listen_port=1,
                inventory_report_interval_ms=0,
                capacity_low_watermark_percent=capacity_low_watermark_percent,
            ),
            KVCRBindings(
                pinning.request_pin,
                pinning.poll_pin_results,
                pinning.release_pin,
                inventory_sink=inventory_sink,
                capacity_needed_callback=capacity_needed_callback,
                policy=policy,
            ),
            KVCRBackendConfigs(
                local_dram=LocalDramInfo(
                    ctypes.addressof(local), len(local), slot_count
                )
            ),
        )
    _OPEN_KVCRS.append(kvcr)
    return kvcr


class _RecordingFIFOPolicy(FIFOPolicy):
    def __init__(self):
        self.scored = []
        self.decided = []
        self.ingested = []
        self.removed = []

    def eviction_score(self, meta, source):
        self.scored.append((meta, source))
        return super().eviction_score(meta, source)

    def decide_eviction(self, meta, source):
        self.decided.append((meta, source))
        return super().decide_eviction(meta, source)

    def on_ingest(self, meta, source):
        self.ingested.append((meta, source))

    def on_remove(self, meta):
        self.removed.append(meta)


class _MatchingHintAdapter:
    def matches(self, key, hint):
        return (
            isinstance(hint, KvSourceLocationsHint)
            and b"test-hash" in hint.block_hashes
        )


def _recovered_record(*, g2: int | None = None, g3: int | None = None) -> _BlockRecord:
    """A block record as recovery rebuilds one: settled residencies, nothing live."""
    return _BlockRecord(
        local_dram=(
            None if g2 is None else _LocalDramResidency(g2, _LocalDramState.READY)
        ),
        g3=None if g3 is None else _G3Residency(g3),
    )
