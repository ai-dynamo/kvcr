# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import os
import select
import socket
from pathlib import Path
from unittest.mock import Mock

import msgspec
import pytest

from kvcr import guard_protocol as protocol_module
from kvcr.config import G3Options, LocalDramInfo
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

_POOL_INDEX = 3
_ROW_STRIDE = 1024
_GENERATION = "a" * 32
_DEVICE = 2049
_INODE = 42
_DIGEST = "opaque digest: leave unchanged"
_JOURNAL_BYTES = 2 * _JOURNAL_HEADER_BYTES
_MAPPING_BYTES = _JOURNAL_BYTES + 8195
_TIER_CONFIG = _TierConfig(_ROW_STRIDE, None)


def test_a_pidfd_that_will_not_close_is_given_up_anyway() -> None:
    """The holder is gone whether or not the kernel agrees."""
    liveness = object.__new__(PidfdLiveness)
    liveness._pidfd = 999_999  # never a live descriptor in this process

    liveness.close()

    assert liveness._pidfd == -1
    liveness.close()


def test_a_kernel_without_peer_pidfd_is_told_why() -> None:
    """A supported refusal, not an internal error the operator cannot act on."""
    connection = Mock()
    connection.getsockopt.side_effect = OSError(errno.ENOPROTOOPT, "not supported")

    with pytest.raises(KVCRServiceError, match="Linux 6.5"):
        PidfdLiveness.from_peer_socket(connection)


def test_pidfd_is_derived_from_the_accepted_peer_socket() -> None:
    accepted, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        liveness = PidfdLiveness.from_peer_socket(accepted)
        poller = select.poll()
        poller.register(liveness.fileno(), select.POLLIN)
        assert poller.poll(0) == []

        liveness.close()
        liveness.close()
        with pytest.raises(ValueError, match="pidfd is closed"):
            liveness.fileno()
    finally:
        accepted.close()
        peer.close()


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
    pool_index: int = _POOL_INDEX,
    mapping_bytes: int = _MAPPING_BYTES,
    tier_config: _TierConfig = _TIER_CONFIG,
) -> _Granted:
    return _Granted(
        pool_index,
        KVCRPoolSpec(
            pool_id=f"pool_{_POOL_INDEX}",
            path=f"/tmp/kvcr-pool_{_POOL_INDEX}-{_GENERATION}",
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


def test_claim_and_release_use_typed_messages_and_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    connection = _RecordingConnection(
        [_grant(), _Released(1)],
        events,
    )
    attachment = _Attachment(events)
    attach = Mock(return_value=attachment)
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)

    hold = KVCRClient("/unused").claim(
        _POOL_INDEX, _ROW_STRIDE, _DIGEST, ("127.0.0.1", 5555)
    )

    assert connection.sent == [
        _Claim(_POOL_INDEX, _DIGEST, _TIER_CONFIG, "127.0.0.1", 5555, 1)
    ]
    assert msgspec.to_builtins(connection.sent[0]) == {
        "type": "claim",
        "pool_index": _POOL_INDEX,
        "compatibility_digest": _DIGEST,
        "tier_config": {"row_stride": _ROW_STRIDE, "g3": None},
        "control_host": "127.0.0.1",
        "control_port": 5555,
        "version": 1,
    }
    assert msgspec.to_builtins(_grant()) == {
        "type": "granted",
        "pool_index": _POOL_INDEX,
        "spec": {
            "pool_id": f"pool_{_POOL_INDEX}",
            "path": f"/tmp/kvcr-pool_{_POOL_INDEX}-{_GENERATION}",
            "generation": _GENERATION,
            "device": _DEVICE,
            "inode": _INODE,
            "mapping_bytes": _MAPPING_BYTES,
            "journal_bytes": _JOURNAL_BYTES,
        },
        "tier_config": {"row_stride": _ROW_STRIDE, "g3": None},
        "version": 1,
    }
    attach.assert_called_once_with(_grant().spec)
    assert hold.local_dram == LocalDramInfo(1234 + _JOURNAL_BYTES, 8192, 8)
    # The endpoint a Guard will answer on, handed over with the grant.
    assert hold._control_listener_fd == connection.handed_fd

    hold.release()

    # Released rather than disowned, so the hold closes what it was given.
    assert connection.sent_fds == []

    assert connection.sent[-1] == _Release(1)
    assert msgspec.to_builtins(connection.sent[-1]) == {
        "type": "release",
        "version": 1,
    }
    assert msgspec.to_builtins(_Released(1)) == {
        "type": "released",
        "version": 1,
    }
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


def test_g3_options_round_trip_through_the_claim_wire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    g3 = G3Options(
        paths=(tmp_path / "g3",),
        capacity_bytes_per_file=8192,
        backend="FILE",
        backend_options={"mode": "direct"},
    )
    encoded_g3 = _G3Config(
        paths=(str(g3.paths[0]),),
        capacity_bytes_per_file=8192,
        backend="FILE",
        backend_options={"mode": "direct"},
    )
    tier_config = _TierConfig(_ROW_STRIDE, encoded_g3)
    connection = _RecordingConnection([_grant(tier_config=tier_config), _Released(1)])
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(
        protocol_module.KVCRPoolAttachment,
        "attach",
        Mock(return_value=_Attachment()),
    )

    KVCRClient("/unused").claim(
        _POOL_INDEX, _ROW_STRIDE, _DIGEST, ("127.0.0.1", 5555), g3
    ).release()
    encoded = msgspec.to_builtins(connection.sent[0])["tier_config"]["g3"]

    assert encoded == {
        "paths": (str(g3.paths[0]),),
        "capacity_bytes_per_file": 8192,
        "backend": "FILE",
        "backend_options": {"mode": "direct"},
    }
    assert connection.sent[0].tier_config == tier_config


@pytest.mark.parametrize(
    "grant",
    [
        _grant(pool_index=_POOL_INDEX + 1),
        _grant(tier_config=_TierConfig(_ROW_STRIDE * 2, None)),
        _grant(mapping_bytes=_JOURNAL_BYTES + _ROW_STRIDE - 1),
    ],
)
def test_mismatched_grant_is_released_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
    grant: _Granted,
) -> None:
    connection = _RecordingConnection([grant, _Released(1)])
    attach = Mock()
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)

    with pytest.raises(KVCRGuardProtocolError):
        KVCRClient("/unused").claim(
            _POOL_INDEX, _ROW_STRIDE, _DIGEST, ("127.0.0.1", 5555)
        )

    attach.assert_not_called()
    assert connection.sent == [
        _Claim(_POOL_INDEX, _DIGEST, _TIER_CONFIG, "127.0.0.1", 5555, 1),
        _Release(1),
    ]
    assert connection.closed is True


def test_typed_grant_decode_failure_is_released_without_masking_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_error = KVCRGuardProtocolError("invalid granted message")
    connection = _RecordingConnection([decode_error, _Released(1)])
    _connect_with(monkeypatch, connection)
    attach = Mock()
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)

    with pytest.raises(KVCRGuardProtocolError) as raised:
        KVCRClient("/unused").claim(
            _POOL_INDEX, _ROW_STRIDE, _DIGEST, ("127.0.0.1", 5555)
        )

    assert raised.value is decode_error
    attach.assert_not_called()
    assert connection.sent == [
        _Claim(_POOL_INDEX, _DIGEST, _TIER_CONFIG, "127.0.0.1", 5555, 1),
        _Release(1),
    ]
    assert connection.closed is True


def test_mapping_failure_rollback_does_not_mask_the_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping_error = PermissionError("mapping failed")
    connection = _RecordingConnection(
        [_grant(), ConnectionResetError("rollback failed")]
    )
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(
        protocol_module.KVCRPoolAttachment,
        "attach",
        Mock(side_effect=mapping_error),
    )

    with pytest.raises(PermissionError, match="mapping failed") as raised:
        KVCRClient("/unused").claim(
            _POOL_INDEX, _ROW_STRIDE, _DIGEST, ("127.0.0.1", 5555)
        )

    assert raised.value is mapping_error
    assert connection.sent[-1] == _Release(1)
    assert connection.closed is True


def test_unmap_failure_keeps_connection_for_a_later_release() -> None:
    events: list[str] = []
    unmap_error = BufferError("mapping is exported")
    attachment = _Attachment(events, close_error=unmap_error)
    connection = _RecordingConnection([_Released(1)], events)
    hold = KVCRPoolHold(
        local_dram=LocalDramInfo(1234, 8192, 8),
        _attachment=attachment,
        _connection=connection,
    )

    with pytest.raises(BufferError, match="mapping is exported") as raised:
        hold.release()

    assert raised.value is unmap_error
    assert connection.sent == []
    assert connection.closed is False
    assert events == ["attachment.close"]

    attachment._close_error = None
    hold.release()

    assert connection.sent == [_Release(1)]
    assert connection.closed is True


def test_release_socket_failure_is_reported_and_connection_is_closed() -> None:
    events: list[str] = []
    connection = _RecordingConnection(
        [ConnectionResetError("release acknowledgement was lost")], events
    )
    hold = KVCRPoolHold(
        local_dram=LocalDramInfo(1234, 8192, 8),
        _attachment=_Attachment(events),
        _connection=connection,
    )

    with pytest.raises(KVCRSocketError, match="acknowledgement was lost"):
        hold.release()

    assert events == [
        "attachment.close",
        "send",
        "receive",
        "connection.close",
    ]
