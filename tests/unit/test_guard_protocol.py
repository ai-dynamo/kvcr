# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import os
import select
import socket
from unittest.mock import MagicMock, Mock

import msgspec
import pytest

from kvcr import guard_protocol as protocol_module
from kvcr.config import LocalDramOptions
from kvcr.control_channels import (
    KVCRGuardProtocolError,
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
    _Release,
    _Released,
    _TierConfig,
)
from kvcr.memory import _JOURNAL_HEADER_BYTES, KVCRPoolSpec

_GUARD_INDEX = 3
_ROW_STRIDE = 1024
_GENERATION = "a" * 32
_DEVICE = 2049
_INODE = 42
_DIGEST = "opaque digest: leave unchanged"
_JOURNAL_BYTES = 2 * _JOURNAL_HEADER_BYTES
_MAPPING_BYTES = _JOURNAL_BYTES + 8195
_TIER_CONFIG = _TierConfig(_ROW_STRIDE, None)


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
        liveness = PidfdLiveness.from_peer_socket(accepted)
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

    # A kernel without SO_PEERPIDFD gets a supported refusal, not an internal
    # error the operator cannot act on.
    unsupported = Mock()
    unsupported.getsockopt.side_effect = OSError(errno.ENOPROTOOPT, "not supported")
    with pytest.raises(KVCRServiceError, match="Linux 6.5"):
        PidfdLiveness.from_peer_socket(unsupported)


class _RecordingConnection:
    def __init__(
        self,
        responses: list[object | BaseException],
        events: list[str] | None = None,
    ) -> None:
        self.responses = responses
        self.events = events if events is not None else []
        self.sent: list[object] = []
        self.sent_fds: list[int] = []
        # A real descriptor, because the claim path closes what it is given.
        self.received_fd: int | None = os.open(os.devnull, os.O_RDONLY)
        self.handed_fd: int | None = None
        self.closed = False

    def send(self, message: object) -> None:
        self.events.append("send")
        self.sent.append(message)

    def send_with_fd(self, message: object, file_descriptor: int) -> None:
        self.send(message)
        self.sent_fds.append(file_descriptor)

    def receive_with_fd(self, decoder: object) -> tuple[object, int | None]:
        # Handing it over transfers ownership, exactly as the real channel does.
        message = self.receive(decoder)
        self.handed_fd, self.received_fd = self.received_fd, None
        return message, self.handed_fd

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
        if self.received_fd is not None:
            os.close(self.received_fd)
            self.received_fd = None


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

    @property
    def data_address(self) -> int:
        return self.address + _JOURNAL_BYTES

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
        1,
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


def test_g3_terms_no_claimant_could_open_are_refused_at_decode() -> None:
    """The first claim fixes a pool's tiers forever, so terms the claimant's
    G3 would reject must be refused before they bind."""
    page = os.sysconf("SC_PAGE_SIZE")
    good = {
        "paths": ("/g3/a",),
        "capacity_bytes_per_file": page,
        "backend": "FILE",
        "backend_options": {},
    }
    with pytest.raises(ValueError, match="page aligned"):
        _TierConfig(page // 2, _G3Config(**good))
    with pytest.raises(ValueError, match="complete slots"):
        _TierConfig(page, _G3Config(**{**good, "capacity_bytes_per_file": page + 1}))
    with pytest.raises(ValueError, match="unique"):
        _TierConfig(page, _G3Config(**{**good, "paths": ("/g3/a", "/g3//a")}))
    _TierConfig(page, _G3Config(**good))


def test_claim_and_release_round_trip_typed_messages_and_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claim/release round-trips typed wire messages, geometry, and ownership."""
    events: list[str] = []
    connection = _RecordingConnection([_grant(), _Released(1)], events)
    attachment = _Attachment(events)
    attach = Mock(return_value=attachment)
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)

    hold = KVCRClient("/unused").claim(
        _GUARD_INDEX, _ROW_STRIDE, _DIGEST, ("127.0.0.1", 5555)
    )

    assert msgspec.to_builtins(connection.sent[0]) == {
        "type": "claim",
        "guard_index": _GUARD_INDEX,
        "compatibility_digest": _DIGEST,
        "tier_config": {
            "row_stride": _ROW_STRIDE,
            "g3": None,
            "remote_fw_dram_backend": "UCX",
        },
        "control_host": "127.0.0.1",
        "control_port": 5555,
        "version": 1,
    }
    assert msgspec.to_builtins(_grant()) == {
        "type": "granted",
        "guard_index": _GUARD_INDEX,
        "spec": {
            "pool_id": f"pool_{_GUARD_INDEX}",
            "path": f"/tmp/kvcr-pool_{_GUARD_INDEX}-{_GENERATION}",
            "generation": _GENERATION,
            "device": _DEVICE,
            "inode": _INODE,
            "mapping_bytes": _MAPPING_BYTES,
            "journal_bytes": _JOURNAL_BYTES,
        },
        "tier_config": {
            "row_stride": _ROW_STRIDE,
            "g3": None,
            "remote_fw_dram_backend": "UCX",
        },
        "version": 1,
    }
    attach.assert_called_once_with(_grant().spec)
    assert hold.local_dram == LocalDramOptions(1234 + _JOURNAL_BYTES, 8192, 8)
    # The endpoint a Guard will answer on, handed over with the grant.
    assert hold._control_listener_fd == connection.handed_fd

    hold.release()

    # Released rather than disowned, so the hold closes what it was given.
    assert connection.sent_fds == []
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
            _grant(tier_config=_TierConfig(_ROW_STRIDE * 2, None)),
            None,
            id="wrong-stride",
        ),
        pytest.param(
            _grant(mapping_bytes=_JOURNAL_BYTES + _ROW_STRIDE - 1),
            None,
            id="short-mapping",
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
            _GUARD_INDEX, _ROW_STRIDE, _DIGEST, ("127.0.0.1", 5555)
        )

    if original is not None:
        assert raised.value is original
    # A mismatched or undecodable grant is refused before the pool is mapped.
    assert attach.call_count == (1 if mapping_error else 0)
    assert connection.sent == [
        _Claim(_GUARD_INDEX, _DIGEST, _TIER_CONFIG, "127.0.0.1", 5555, 1),
        # Unactivated: this claim never served, so the Guard may resume.
        _Release(1, activated=False),
    ]
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
        local_dram=LocalDramOptions(1234, 8192, 8),
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
