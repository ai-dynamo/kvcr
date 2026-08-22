# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The wire between a KVCR worker and KVCR-Service, and the client half."""

import contextlib
import errno
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Annotated

import msgspec

from .config import LocalDramInfo
from .control_channels import (
    FramedConnection,
    KVCRGuardProtocolError,
    KVCRServiceError,
    KVCRSocketError,
)
from .memory import KVCRPoolAttachment, KVCRPoolSpec

logger = logging.getLogger(__name__)


# SO_PEERPIDFD requires Linux 6.5 or later.
_SO_PEERPIDFD_FALLBACK = 77
_SO_PEERPIDFD = getattr(socket, "SO_PEERPIDFD", _SO_PEERPIDFD_FALLBACK)


class _Claim(msgspec.Struct, frozen=True, tag="claim"):
    pool_index: Annotated[int, msgspec.Meta(ge=0)]
    row_stride: Annotated[int, msgspec.Meta(gt=0)]
    compatibility_digest: str


class _Release(msgspec.Struct, frozen=True, tag="release"):
    pass


class _Granted(msgspec.Struct, frozen=True, tag="granted"):
    pool_index: int
    spec: KVCRPoolSpec


class _Released(msgspec.Struct, frozen=True, tag="released"):
    pass


class _Error(msgspec.Struct, frozen=True, tag="error"):
    message: str


_CLAIM_DECODER = msgspec.msgpack.Decoder(_Claim)
_RELEASE_DECODER = msgspec.msgpack.Decoder(_Release)
_CLAIM_RESPONSE_DECODER = msgspec.msgpack.Decoder(_Granted | _Error)
_RELEASE_RESPONSE_DECODER = msgspec.msgpack.Decoder(_Released | _Error)


class PidfdLiveness:
    """Own the pidfd returned for an accepted Unix-socket peer."""

    def __init__(self, pidfd: int) -> None:
        self._pidfd = pidfd

    @classmethod
    def from_peer_socket(cls, connection: socket.socket) -> "PidfdLiveness":
        try:
            pidfd = connection.getsockopt(socket.SOL_SOCKET, _SO_PEERPIDFD)
        except OSError as error:
            if error.errno != errno.ENOPROTOOPT:
                raise
            message = "SO_PEERPIDFD requires Linux 6.5 or later"
            logger.warning(message)
            raise RuntimeError(message) from error
        return cls(pidfd)

    def fileno(self) -> int:
        if self._pidfd < 0:
            raise ValueError("pidfd is closed")
        return self._pidfd

    def close(self) -> None:
        pidfd, self._pidfd = self._pidfd, -1
        if pidfd >= 0:
            os.close(pidfd)


@dataclass
class KVCRPoolHold:
    """A mapped pool and the connection holding its lease."""

    local_dram: LocalDramInfo
    _attachment: KVCRPoolAttachment
    _connection: FramedConnection
    _release_attempted: bool = field(default=False, init=False, repr=False)

    def release(self) -> None:
        """Stop local access before releasing the connection-scoped lease."""
        if self._release_attempted:
            return
        self._attachment.close()
        self._release_attempted = True

        try:
            _send_release(self._connection)
            self._connection.close()
        except BaseException as error:
            _close_quietly(self._connection)
            if isinstance(error, (OSError, EOFError)):
                raise KVCRSocketError(
                    f"KVCR-Service release failed: {error}"
                ) from error
            raise


class KVCRClient:
    """Synchronous client for a standalone KVCR service."""

    def __init__(self, socket_path: str | os.PathLike[str]) -> None:
        self._socket_path = os.fspath(socket_path)

    def claim(
        self,
        pool_index: int,
        row_stride: int,
        compatibility_digest: str,
    ) -> KVCRPoolHold:
        """Claim and map one service-owned pool."""
        request = msgspec.convert(
            {
                "pool_index": pool_index,
                "row_stride": row_stride,
                "compatibility_digest": compatibility_digest,
            },
            type=_Claim,
        )
        connection = FramedConnection.connect(self._socket_path)
        attachment: KVCRPoolAttachment | None = None
        release_needed = False
        grant_received = False
        try:
            connection.send(request)
            release_needed = True
            response = connection.receive(_CLAIM_RESPONSE_DECODER)
            if isinstance(response, _Error):
                release_needed = False
                raise KVCRServiceError(response.message)
            grant_received = True
            spec = _grant_spec(response, pool_index, row_stride)
            attachment = KVCRPoolAttachment.attach(spec)
            return KVCRPoolHold(
                local_dram=LocalDramInfo(
                    attachment.address,
                    spec.effective_bytes,
                    spec.rows,
                ),
                _attachment=attachment,
                _connection=connection,
            )
        except BaseException as error:
            local_access_stopped = True
            if attachment is not None:
                try:
                    attachment.close()
                except BaseException:
                    local_access_stopped = False
            if release_needed and local_access_stopped:
                with contextlib.suppress(BaseException):
                    _send_release(connection)
            _close_quietly(connection)
            if not grant_received and isinstance(error, (OSError, EOFError)):
                raise KVCRSocketError(f"KVCR-Service claim failed: {error}") from error
            raise


def _grant_spec(
    response: _Granted,
    requested_pool_index: int,
    requested_row_stride: int,
) -> KVCRPoolSpec:
    if response.pool_index != requested_pool_index:
        raise KVCRGuardProtocolError(
            "claim pool mismatch: "
            f"requested {requested_pool_index}, got {response.pool_index}"
        )
    spec = response.spec
    if spec.row_stride != requested_row_stride:
        raise KVCRGuardProtocolError(
            "claim geometry mismatch: "
            f"requested row_stride={requested_row_stride}, got {spec.row_stride}"
        )
    return spec


def _send_release(connection: FramedConnection) -> None:
    connection.send(_Release())
    response = connection.receive(_RELEASE_RESPONSE_DECODER)
    if isinstance(response, _Error):
        raise KVCRServiceError(response.message)


def _close_quietly(connection: FramedConnection) -> None:
    with contextlib.suppress(BaseException):
        connection.close()
