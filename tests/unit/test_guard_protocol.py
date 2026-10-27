# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import mmap
import os
import select
import socket
from typing import Any
from unittest.mock import MagicMock, Mock

import msgspec
import pytest
from _kvcr_test_utils import _wait_until

from kvcr import guard_protocol as protocol_module
from kvcr.config import LocalDramOptions
from kvcr.control_channels import (
    KVCRGuardProtocolError,
    KVCRMsgFramingError,
    KVCRServiceError,
    KVCRSocketError,
)
from kvcr.guard_protocol import (
    KVCRClient,
    KVCRPoolHold,
    PidfdLiveness,
    _Claim,
    _Error,
    _G3Config,
    _Granted,
    _Heartbeat,
    _PoolDescriptor,
    _Release,
    _Released,
    _TierConfig,
)
from kvcr.memory import _JOURNAL_HEADER_BYTES, KVCRPoolSpec

_GUARD_INDEX = 3
_POOL_LAYOUTS = [("pool0", 1024), ("pool1", 3072)]
_POOL_SIZES = (mmap.PAGESIZE, 2 * mmap.PAGESIZE)
_GENERATION = "a" * 32
_DEVICE = 2049
_INODE = 42
_DIGEST = "opaque digest: leave unchanged"
_JOURNAL_BYTES = 2 * _JOURNAL_HEADER_BYTES
_POOL_OFFSETS = (_JOURNAL_BYTES, _JOURNAL_BYTES + _POOL_SIZES[0])
_MAPPING_BYTES = _JOURNAL_BYTES + sum(_POOL_SIZES)
_TIER_CONFIG = _TierConfig(_POOL_LAYOUTS, None)
_WIRE_POOLS = tuple(
    _PoolDescriptor(name, size_bytes, block_size_bytes, offset_bytes)
    for (name, block_size_bytes), size_bytes, offset_bytes in zip(
        _POOL_LAYOUTS, _POOL_SIZES, _POOL_OFFSETS, strict=True
    )
)


def test_close_swaps_the_pidfd_under_its_lock() -> None:
    """Shutdown can race a failed claim's cleanup here; an unsynchronized swap
    lets both threads close, and the second can hit a reused descriptor."""

    liveness = PidfdLiveness(os.dup(0))
    lock = MagicMock()
    liveness._close_lock = lock

    liveness.close()
    liveness.close()

    # Both closes took the lock; only the first found a descriptor to close.
    assert lock.__enter__.call_count == 2
    assert liveness._pidfd == -1


def test_a_peer_pidfd_serves_polling_until_closed_then_refuses_use() -> None:
    """A peer pidfd polls until closed, then is refused; old kernels get why."""
    accepted, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        liveness = PidfdLiveness.from_peer_socket(accepted, os.pidfd_open(os.getpid()))
        poller = select.poll()
        poller.register(liveness.fileno(), select.POLLIN)
        assert poller.poll(0) == []  # the holder (this process) is still alive

        liveness.close()
        liveness.close()
        with pytest.raises(ValueError, match="pidfd is closed"):
            liveness.fileno()
    finally:
        accepted.close()
        peer.close()

    # A pidfd the kernel will not close is given up anyway: the holder is
    # gone whether or not the kernel agrees.
    stubborn = PidfdLiveness(999_999)  # never a live descriptor in this process
    stubborn.close()
    with pytest.raises(ValueError, match="pidfd is closed"):
        stubborn.fileno()
    stubborn.close()

    # A legacy kernel still requires the worker to supply its own pidfd.
    unsupported = Mock()
    unsupported.getsockopt.side_effect = OSError(errno.ENOPROTOOPT, "not supported")
    with pytest.raises(KVCRServiceError, match="worker pidfd is required"):
        PidfdLiveness.from_peer_socket(unsupported)


class _RecordingConnection:
    def __init__(
        self,
        responses: list[object | BaseException],
        events: list[str] | None = None,
        received_fd_count: int = 1,
    ) -> None:
        self.responses = responses
        self.events = events if events is not None else []
        self.sent: list[object] = []
        self.sent_fds: list[int] = []
        # A real descriptor, because the claim path closes what it is given.
        self.received_fds = [
            os.open(os.devnull, os.O_RDONLY) for _ in range(received_fd_count)
        ]
        self.handed_fds: list[int] = []
        self.closed = False

    def send(self, message: object) -> None:
        self.events.append("send")
        self.sent.append(message)

    def send_with_fd(self, message: object, *file_descriptors: int) -> None:
        self.send(message)
        self.sent_fds.extend(file_descriptors)

    def receive_with_fd(
        self, decoder: object, expected_counts: Any
    ) -> tuple[object, list[int]]:
        message = self.receive(decoder)
        if len(self.received_fds) not in expected_counts(message):
            # Refused deliveries close what they received, as the real channel does.
            self._close_received()
            raise KVCRMsgFramingError("unexpected number of file descriptors")
        # Handing them over transfers ownership, exactly as the real channel does.
        self.handed_fds, self.received_fds = self.received_fds, []
        return message, self.handed_fds

    def _close_received(self) -> None:
        for file_descriptor in self.received_fds:
            os.close(file_descriptor)
        self.received_fds = []

    def receive(self, _decoder: object) -> object:
        self.events.append("receive")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self) -> None:
        self.events.append("connection.close")
        self.closed = True
        # Whatever the claim never took off our hands, exactly like a real one.
        self._close_received()


class _Attachment:
    def __init__(
        self,
        events: list[str] | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._close_error = close_error

    @property
    def address(self) -> int:
        return 1234

    def close(self) -> None:
        if self._events is not None:
            self._events.append("attachment.close")
        if self._close_error is not None:
            raise self._close_error


def _grant(
    *,
    guard_index: int = _GUARD_INDEX,
    mapping_bytes: int = _MAPPING_BYTES,
    tier_config: _TierConfig = _TIER_CONFIG,
    pools: tuple[_PoolDescriptor, ...] = _WIRE_POOLS,
) -> _Granted:
    return _Granted(
        guard_index,
        KVCRPoolSpec(
            pool_id=f"pool_{_GUARD_INDEX}",
            path=f"/tmp/kvcr-pool_{_GUARD_INDEX}-{_GENERATION}",
            generation=_GENERATION,
            device=_DEVICE,
            inode=_INODE,
            mapping_bytes=mapping_bytes,
            journal_bytes=_JOURNAL_BYTES,
        ),
        tier_config,
        pools,
        1,
    )


def _grant_with_pool(index: int = 0, **changes) -> _Granted:
    pools = list(_WIRE_POOLS)
    pools[index] = msgspec.structs.replace(pools[index], **changes)
    return _grant(pools=tuple(pools))


def _local_dram(address: int = 1234) -> LocalDramOptions:
    return LocalDramOptions(
        [
            (pool.name, address + pool.offset_bytes, pool.size_bytes)
            for pool in _WIRE_POOLS
        ]
    )


def _connect_with(
    monkeypatch: pytest.MonkeyPatch,
    connection: _RecordingConnection,
) -> None:
    monkeypatch.setattr(
        protocol_module.FramedConnection,
        "connect",
        lambda _endpoint: connection,
    )


def test_pool_descriptor_constraints_are_part_of_the_wire_contract() -> None:
    with pytest.raises(ValueError, match="positive"):
        _PoolDescriptor("pool", 0, 1)
    with pytest.raises(ValueError, match="complete KV block"):
        _PoolDescriptor("pool", 1023, 1024)


def test_g3_config_validates_paths_and_tier_capacity() -> None:
    good = {
        "paths": ("/g3/a",),
        "capacity_bytes_per_file": 1,
        "backend": "FILE",
        "backend_options": {},
    }
    with pytest.raises(ValueError, match="absolute"):
        _G3Config(**{**good, "paths": ("g3/a",)})
    with pytest.raises(ValueError, match="unique"):
        _G3Config(**{**good, "paths": ("/g3/a", "/g3//a")})

    with pytest.raises(ValueError, match="does not support multiple pools"):
        _TierConfig(
            _POOL_LAYOUTS,
            _G3Config(**{**good, "capacity_bytes_per_file": mmap.PAGESIZE}),
        )
    with pytest.raises(ValueError, match="at least one complete slot"):
        _TierConfig(
            [("pool", mmap.PAGESIZE)],
            _G3Config(**{**good, "capacity_bytes_per_file": mmap.PAGESIZE - 1}),
        )
    with pytest.raises(ValueError, match="page-aligned"):
        _TierConfig(
            [("pool", mmap.PAGESIZE // 2)],
            _G3Config(**{**good, "capacity_bytes_per_file": mmap.PAGESIZE}),
        )

    tier = _TierConfig(
        [("pool", mmap.PAGESIZE)],
        _G3Config(**{**good, "capacity_bytes_per_file": 2 * mmap.PAGESIZE - 1}),
    )
    assert (
        msgspec.msgpack.decode(msgspec.msgpack.encode(tier), type=_TierConfig) == tier
    )


def test_claim_and_release_round_trip_typed_messages_and_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claim/release round-trips typed wire messages, geometry, and ownership."""
    events: list[str] = []
    grant = msgspec.structs.replace(_grant(), dead_incarnations=("dead-primary",))
    decoded = protocol_module._CLAIM_RESPONSE_DECODER.decode(
        msgspec.msgpack.encode(grant)
    )
    connection = _RecordingConnection([decoded, _Released(1)], events)
    attachment = _Attachment(events)
    attach = Mock(return_value=attachment)
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)

    hold = KVCRClient("/unused").claim(
        _GUARD_INDEX, _POOL_LAYOUTS, _DIGEST, ("127.0.0.1", 5555)
    )

    assert hold._incarnation
    assert hold._dead_incarnations == ("dead-primary",)
    assert msgspec.to_builtins(connection.sent[0]) == {
        "type": "claim",
        "guard_index": _GUARD_INDEX,
        "compatibility_digest": _DIGEST,
        "tier_config": {
            "pool_layouts": _POOL_LAYOUTS,
            "g3": None,
            "remote_fw_dram_backend": "UCX",
        },
        "control_host": "127.0.0.1",
        "control_port": 5555,
        "version": 1,
        "incarnation": hold._incarnation,
        "attach_mode": "path",
    }
    grant_wire = msgspec.to_builtins(grant)
    assert grant_wire["dead_incarnations"] == ("dead-primary",)
    assert grant_wire["type"] == "granted"
    assert grant_wire["version"] == 1
    assert grant_wire["guard_index"] == _GUARD_INDEX
    assert grant_wire["tier_config"] == {
        "pool_layouts": _POOL_LAYOUTS,
        "g3": None,
        "remote_fw_dram_backend": "UCX",
    }
    assert grant_wire["pools"] == msgspec.to_builtins(_WIRE_POOLS)
    attach.assert_called_once_with(grant.spec)
    assert hold.local_dram == _local_dram()
    # The endpoint a Guard will answer on, handed over with the grant.
    assert hold._control_listener_fd == connection.handed_fds[-1]

    hold.release()

    # Released rather than disowned, so the hold closes what it was given.
    assert len(connection.sent_fds) == 1  # The claim carried the worker pidfd.
    assert msgspec.to_builtins(connection.sent[-1]) == {
        "type": "release",
        "version": 1,
        "activated": True,
    }
    assert msgspec.to_builtins(_Released(1)) == {"type": "released", "version": 1}
    assert msgspec.to_builtins(_Error("failure", 1)) == {
        "type": "error",
        "message": "failure",
        "version": 1,
    }
    assert events == [
        "send",
        "receive",
        "attachment.close",
        "send",
        "receive",
        "connection.close",
    ]


@pytest.mark.parametrize(
    ("reply", "mapping_error"),
    [
        pytest.param(_grant(guard_index=_GUARD_INDEX + 1), None, id="wrong-guard"),
        pytest.param(
            _grant(
                tier_config=_TierConfig(
                    [("pool0", _POOL_LAYOUTS[0][1] * 2), _POOL_LAYOUTS[1]], None
                )
            ),
            None,
            id="wrong-tier-layout",
        ),
        pytest.param(_grant(pools=_WIRE_POOLS[:1]), None, id="wrong-pool-count"),
        pytest.param(
            _grant_with_pool(block_size_bytes=_POOL_LAYOUTS[0][1] * 2),
            None,
            id="wrong-pool-block-size",
        ),
        pytest.param(
            _grant_with_pool(1, offset_bytes=_POOL_OFFSETS[1] + mmap.PAGESIZE),
            None,
            id="noncontiguous-offset",
        ),
        pytest.param(_grant_with_pool(size_bytes=_POOL_SIZES[0] + 1), None, id="size"),
        pytest.param(
            _grant(mapping_bytes=_MAPPING_BYTES + mmap.PAGESIZE), None, id="mapping"
        ),
        pytest.param(
            KVCRGuardProtocolError("invalid granted message"),
            None,
            id="undecodable-grant",
        ),
        pytest.param(_grant(), PermissionError("mapping failed"), id="mapping-failed"),
    ],
)
def test_a_failed_claim_is_released_without_masking_the_original(
    monkeypatch: pytest.MonkeyPatch,
    reply: _Granted | BaseException,
    mapping_error: BaseException | None,
) -> None:
    """A failed claim hands the pool back and raises the original, unmasked error."""
    # A rollback that itself fails must not mask the mapping error either.
    rollback_reply: object = (
        ConnectionResetError("rollback failed") if mapping_error else _Released(1)
    )
    connection = _RecordingConnection([reply, rollback_reply])
    attach = Mock(side_effect=mapping_error)
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)

    original = mapping_error or (reply if isinstance(reply, BaseException) else None)
    with pytest.raises(
        type(original) if original is not None else KVCRGuardProtocolError
    ) as raised:
        KVCRClient("/unused").claim(
            _GUARD_INDEX, _POOL_LAYOUTS, _DIGEST, ("127.0.0.1", 5555)
        )

    if original is not None:
        assert raised.value is original
    # A mismatched or undecodable grant is refused before the pool is mapped.
    assert attach.call_count == (1 if mapping_error else 0)
    claim, release = connection.sent
    assert isinstance(claim, _Claim)
    # Unactivated: this claim never served, so the Guard may resume.
    assert release == _Release(1, activated=False)
    assert connection.closed is True


def test_release_failures_leave_a_retry_and_report_a_lost_acknowledgement() -> None:
    """An unmap failure leaves the lease retryable; a lost release ack is reported."""
    events: list[str] = []
    unmap_error = BufferError("mapping is exported")
    attachment = _Attachment(events, close_error=unmap_error)
    connection = _RecordingConnection(
        [ConnectionResetError("release acknowledgement was lost")], events
    )
    hold = KVCRPoolHold(
        local_dram=_local_dram(attachment.address),
        _pools=_WIRE_POOLS,
        _attachment=attachment,
        _connection=connection,
    )

    with pytest.raises(BufferError, match="mapping is exported") as raised:
        hold.release()

    # The lease is untouched: nothing sent, connection open for a later release.
    assert raised.value is unmap_error
    assert connection.sent == []
    assert connection.closed is False
    assert events == ["attachment.close"]

    attachment._close_error = None

    with pytest.raises(KVCRSocketError, match="acknowledgement was lost"):
        hold.release()

    # The retry proceeded, and the lost ack still surrendered the connection.
    assert connection.sent == [_Release(1)]
    assert connection.closed is True
    assert events == [
        "attachment.close",
        "attachment.close",
        "send",
        "receive",
        "connection.close",
    ]


def test_failed_unmap_keeps_heartbeat_armed() -> None:
    attachment = _Attachment(close_error=BufferError("mapping is exported"))
    connection = _RecordingConnection([_Released(1)])
    hold = KVCRPoolHold(
        local_dram=_local_dram(attachment.address),
        _pools=_WIRE_POOLS,
        _attachment=attachment,
        _connection=connection,
        _heartbeat_enabled=True,
    )
    try:
        with pytest.raises(BufferError, match="mapping is exported"):
            hold.release()
        assert not hold._heartbeat_stop.is_set()
        _wait_until(
            lambda: any(isinstance(message, _Heartbeat) for message in connection.sent),
            timeout=1,
        )
    finally:
        attachment._close_error = None
        hold.release()


def test_pool_hold_starts_heartbeat_only_when_granted(monkeypatch) -> None:
    thread = Mock()
    monkeypatch.setattr(protocol_module.threading, "Thread", Mock(return_value=thread))
    kwargs = dict(
        local_dram=_local_dram(),
        _pools=_WIRE_POOLS,
        _attachment=_Attachment(),
        _connection=_RecordingConnection([]),
    )

    KVCRPoolHold(**kwargs)
    thread.start.assert_not_called()
    KVCRPoolHold(**kwargs, _heartbeat_enabled=True)
    thread.start.assert_called_once_with()


def _fd_claim(monkeypatch, response, fd_count):
    """Claim in fd mode against a service that answers with `response`."""
    connection = _RecordingConnection(
        [response, _Released(1)], received_fd_count=fd_count
    )
    attach_fd = Mock(return_value=_Attachment())
    attach = Mock(side_effect=AssertionError("fd mode must not attach by path"))
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach_fd", attach_fd)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)
    return connection, attach_fd


def _fd_grant(**changes):
    return msgspec.structs.replace(_grant(), attach_mode="fd", **changes)


def test_fd_claim_round_trips_and_hands_both_descriptors_over(monkeypatch) -> None:
    connection, attach_fd = _fd_claim(monkeypatch, _fd_grant(), fd_count=2)
    pool_fd, listener_fd = connection.received_fds

    hold = KVCRClient("/unused").claim(
        _GUARD_INDEX, _POOL_LAYOUTS, _DIGEST, ("127.0.0.1", 5555), attach_mode="fd"
    )

    assert connection.sent[0].attach_mode == "fd"
    # The pool descriptor is the attachment's now; the listener stays the hold's.
    attach_fd.assert_called_once_with(pool_fd, _grant().spec)
    assert hold._control_listener_fd == listener_fd
    hold.release()


def test_each_response_states_the_descriptor_count_it_carries() -> None:
    counts = protocol_module._response_fd_counts
    assert counts(_Error("no", 1)) == (0,)
    assert counts(_grant()) == (1,)
    assert counts(_fd_grant()) == (2,)


@pytest.mark.parametrize(
    ("claim_mode", "grant", "fd_count", "error", "match"),
    [
        # A fd grant needs both descriptors.
        ("fd", _fd_grant(), 0, KVCRMsgFramingError, "unexpected number"),
        # A v1 service never heard of attach_mode: it echoes path with one fd.
        ("fd", _grant(), 1, KVCRGuardProtocolError, "attach mode mismatch"),
        # A fd grant answering a path claim.
        ("path", _fd_grant(), 2, KVCRGuardProtocolError, "attach mode mismatch"),
    ],
)
def test_a_grant_that_disagrees_with_the_claim_fails_closed(
    monkeypatch, claim_mode, grant, fd_count, error, match
) -> None:
    connection, attach_fd = _fd_claim(monkeypatch, grant, fd_count)
    received = list(connection.received_fds)

    with pytest.raises(error, match=match):
        KVCRClient("/unused").claim(
            _GUARD_INDEX,
            _POOL_LAYOUTS,
            _DIGEST,
            ("127.0.0.1", 5555),
            attach_mode=claim_mode,
        )

    attach_fd.assert_not_called()
    for file_descriptor in received:
        with pytest.raises(OSError):
            os.fstat(file_descriptor)


def test_v1_messages_without_an_attach_mode_still_decode_as_path() -> None:
    """A v1 peer omits the field; an unknown mode is refused, never guessed."""
    claim = msgspec.to_builtins(
        _Claim(
            _GUARD_INDEX,
            _DIGEST,
            _TierConfig(_POOL_LAYOUTS, None),
            "127.0.0.1",
            5555,
            1,
        )
    )
    v1_claim = {k: v for k, v in claim.items() if k != "attach_mode"}
    assert (
        protocol_module._CLAIM_DECODER.decode(
            msgspec.msgpack.encode(v1_claim)
        ).attach_mode
        == "path"
    )

    v1_grant = {
        k: v for k, v in msgspec.to_builtins(_grant()).items() if k != "attach_mode"
    }
    assert (
        protocol_module._CLAIM_RESPONSE_DECODER.decode(
            msgspec.msgpack.encode(v1_grant)
        ).attach_mode
        == "path"
    )

    with pytest.raises(msgspec.ValidationError):
        protocol_module._CLAIM_DECODER.decode(
            msgspec.msgpack.encode({**claim, "attach_mode": "memfd"})
        )
