# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import select
import socket
from unittest.mock import Mock

import msgspec
import pytest

from kvcr import guard_protocol as protocol_module
from kvcr.config import LocalDramInfo
from kvcr.control_channels import KVCRGuardProtocolError, KVCRSocketError
from kvcr.guard_protocol import (
    KVCRClient,
    KVCRPoolHold,
    PidfdLiveness,
    _Claim,
    _Error,
    _Granted,
    _Release,
    _Released,
)
from kvcr.memory import KVCRPoolSpec

_POOL_INDEX = 3
_ROW_STRIDE = 1024
_GENERATION = "a" * 32
_DEVICE = 2049
_INODE = 42
_DIGEST = "opaque digest: leave unchanged"


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
        self.closed = False

    def send(self, message: object) -> None:
        self.events.append("send")
        self.sent.append(message)

    def receive(self, _decoder: object) -> object:
        self.events.append("receive")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self) -> None:
        self.events.append("connection.close")
        self.closed = True


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
    pool_index: int = _POOL_INDEX,
) -> _Granted:
    return _Granted(
        pool_index,
        KVCRPoolSpec(
            pool_id=f"pool_{_POOL_INDEX}",
            path=f"/tmp/kvcr-pool_{_POOL_INDEX}-{_GENERATION}",
            generation=_GENERATION,
            device=_DEVICE,
            inode=_INODE,
            effective_bytes=8192,
            rows=8,
            row_stride=_ROW_STRIDE,
        ),
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
        [_grant(), _Released()],
        events,
    )
    attachment = _Attachment(events)
    attach = Mock(return_value=attachment)
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)

    hold = KVCRClient("/unused").claim(_POOL_INDEX, _ROW_STRIDE, _DIGEST)

    assert connection.sent == [_Claim(_POOL_INDEX, _ROW_STRIDE, _DIGEST)]
    assert msgspec.to_builtins(connection.sent[0]) == {
        "type": "claim",
        "pool_index": _POOL_INDEX,
        "row_stride": _ROW_STRIDE,
        "compatibility_digest": _DIGEST,
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
            "effective_bytes": 8192,
            "rows": 8,
            "row_stride": _ROW_STRIDE,
        },
    }
    attach.assert_called_once_with(_grant().spec)
    assert hold.local_dram == LocalDramInfo(1234, 8192, 8)

    hold.release()

    assert connection.sent[-1] == _Release()
    assert msgspec.to_builtins(connection.sent[-1]) == {"type": "release"}
    assert msgspec.to_builtins(_Released()) == {"type": "released"}
    assert msgspec.to_builtins(_Error("failure")) == {
        "type": "error",
        "message": "failure",
    }
    assert events == [
        "send",
        "receive",
        "attachment.close",
        "send",
        "receive",
        "connection.close",
    ]


def test_invalid_grant_is_released_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grant = _grant(pool_index=_POOL_INDEX + 1)
    connection = _RecordingConnection([grant, _Released()])
    attach = Mock()
    _connect_with(monkeypatch, connection)
    monkeypatch.setattr(protocol_module.KVCRPoolAttachment, "attach", attach)

    with pytest.raises(KVCRGuardProtocolError):
        KVCRClient("/unused").claim(_POOL_INDEX, _ROW_STRIDE, _DIGEST)

    attach.assert_not_called()
    assert connection.sent == [
        _Claim(_POOL_INDEX, _ROW_STRIDE, _DIGEST),
        _Release(),
    ]
    assert connection.closed is True

    decode_error = KVCRGuardProtocolError("invalid granted message")
    connection = _RecordingConnection([decode_error, _Released()])
    _connect_with(monkeypatch, connection)

    with pytest.raises(KVCRGuardProtocolError) as raised:
        KVCRClient("/unused").claim(_POOL_INDEX, _ROW_STRIDE, _DIGEST)

    assert raised.value is decode_error
    attach.assert_not_called()
    assert connection.sent == [
        _Claim(_POOL_INDEX, _ROW_STRIDE, _DIGEST),
        _Release(),
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
        KVCRClient("/unused").claim(_POOL_INDEX, _ROW_STRIDE, _DIGEST)

    assert raised.value is mapping_error
    assert connection.sent[-1] == _Release()
    assert connection.closed is True


def test_unmap_failure_keeps_connection_for_a_later_release() -> None:
    events: list[str] = []
    unmap_error = BufferError("mapping is exported")
    attachment = _Attachment(events, close_error=unmap_error)
    connection = _RecordingConnection([_Released()], events)
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

    assert connection.sent == [_Release()]
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
