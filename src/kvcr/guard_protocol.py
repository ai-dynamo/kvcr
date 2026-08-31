# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The wire between a KVCR worker and KVCR-Service, and the client half."""

import contextlib
import errno
import logging
import os
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

import msgspec

from .config import G3Options, LocalDramInfo
from .control_channels import (
    FramedConnection,
    KVCRGuardProtocolError,
    KVCRServiceError,
    KVCRSocketError,
)
from .memory import KVCRPoolAttachment, KVCRPoolSpec, _compute_pool_geometry

logger = logging.getLogger(__name__)

ProtocolVersion = Literal[1]
_PROTOCOL_VERSION: ProtocolVersion = 1

# SO_PEERPIDFD requires Linux 6.5 or later.
_SO_PEERPIDFD_FALLBACK = 77
_SO_PEERPIDFD = getattr(socket, "SO_PEERPIDFD", _SO_PEERPIDFD_FALLBACK)


class _G3Config(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    # Constrained here because the claimant that opens G3 under these terms is a
    # replacement started long after this claim was accepted.
    paths: Annotated[tuple[str, ...], msgspec.Meta(min_length=1)]
    capacity_bytes_per_file: Annotated[int, msgspec.Meta(gt=0)]
    backend: Annotated[str, msgspec.Meta(min_length=1)]
    backend_options: dict[str, str]

    def __post_init__(self) -> None:
        if not all(os.path.isabs(path) for path in self.paths):
            raise ValueError("G3 paths must be absolute")


class _TierConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    row_stride: Annotated[int, msgspec.Meta(gt=0)]
    g3: _G3Config | None


class _Claim(msgspec.Struct, frozen=True, tag="claim"):
    pool_index: Annotated[int, msgspec.Meta(ge=0)]
    compatibility_digest: str
    tier_config: _TierConfig
    control_host: str
    control_port: Annotated[int, msgspec.Meta(ge=1, le=65535)]
    version: ProtocolVersion

    def __post_init__(self) -> None:
        # A literal address, because the service binds this holding the lock
        # every other pool needs, and resolving a name there would stall all
        # of them for as long as the resolver takes.
        try:
            socket.inet_pton(socket.AF_INET, self.control_host)
        except OSError as error:
            raise ValueError(
                f"KVCR control address must be a literal IPv4 address: "
                f"{self.control_host}"
            ) from error


class _Release(msgspec.Struct, frozen=True, tag="release"):
    version: ProtocolVersion


class _Granted(msgspec.Struct, frozen=True, tag="granted"):
    pool_index: int
    spec: KVCRPoolSpec
    tier_config: _TierConfig
    version: ProtocolVersion


class _Released(msgspec.Struct, frozen=True, tag="released"):
    version: ProtocolVersion


class _Error(msgspec.Struct, frozen=True, tag="error"):
    message: str
    version: ProtocolVersion


_CLAIM_DECODER = msgspec.msgpack.Decoder(_Claim)
_RELEASE_DECODER = msgspec.msgpack.Decoder(_Release)
_CLAIM_RESPONSE_DECODER = msgspec.msgpack.Decoder(_Granted | _Error)
_RELEASE_RESPONSE_DECODER = msgspec.msgpack.Decoder(_Released | _Error)


class PidfdLiveness:
    """Own the pidfd returned for an accepted Unix-socket peer."""

    def __init__(self, pidfd: int) -> None:
        self._pidfd = pidfd
        self._close_lock = threading.Lock()

    @classmethod
    def from_peer_socket(cls, connection: socket.socket) -> "PidfdLiveness":
        try:
            pidfd = connection.getsockopt(socket.SOL_SOCKET, _SO_PEERPIDFD)
        except OSError as error:
            if error.errno != errno.ENOPROTOOPT:
                raise
            # A refusal with its reason, not an internal error: the kernel is
            # too old for this service and the claimant should be told so.
            message = "SO_PEERPIDFD requires Linux 6.5 or later"
            logger.warning(message)
            raise KVCRServiceError(message) from error
        return cls(pidfd)

    def fileno(self) -> int:
        if self._pidfd < 0:
            raise ValueError("pidfd is closed")
        return self._pidfd

    def close(self) -> None:
        """Give the descriptor up, whether or not the kernel agrees.

        This stops representing a live holder either way, so a failure here is worth a
        log line and nothing else.
        """
        # Under a lock: shutdown can race a failed claim's cleanup here, and
        # an unsynchronized swap lets both threads close -- the second close
        # can reach a descriptor number the process has since reused.
        with self._close_lock:
            pidfd, self._pidfd = self._pidfd, -1
        if pidfd >= 0:
            try:
                os.close(pidfd)
            except OSError:
                # EBADF means this number is already someone else's, so the
                # close may have reached an unrelated descriptor.
                logger.warning("Failed to close a KVCR pidfd", exc_info=True)


@dataclass
class KVCRPoolHold:
    """A mapped pool and the connection holding its lease."""

    local_dram: LocalDramInfo
    _attachment: KVCRPoolAttachment
    _connection: FramedConnection
    _control_listener_fd: int | None = None
    _release_attempted: bool = field(default=False, init=False, repr=False)

    def hand_listener_to(self, adopt: Callable[[int], None]) -> None:
        """Adopt-then-disown: a failed adoption leaves this hold owning the fd,
        which is what closes it on release."""
        if self._control_listener_fd is None:
            return
        adopt(self._control_listener_fd)
        self._control_listener_fd = None

    def release(self) -> None:
        """Stop local access before releasing the connection-scoped lease."""
        if self._release_attempted:
            return
        self._attachment.close()
        if self._control_listener_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._control_listener_fd)
            self._control_listener_fd = None
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
        control_bind: tuple[str, int],
        g3: G3Options | None = None,
    ) -> KVCRPoolHold:
        """Claim and map one service-owned pool."""
        request = _claim_request(
            pool_index, row_stride, compatibility_digest, g3, control_bind
        )
        tier_config = request.tier_config
        connection = FramedConnection.connect(self._socket_path)
        attachment: KVCRPoolAttachment | None = None
        listener_fd: int | None = None
        release_needed = False
        grant_received = False
        try:
            connection.send(request)
            release_needed = True
            response, listener_fd = connection.receive_with_fd(_CLAIM_RESPONSE_DECODER)
            if isinstance(response, _Error):
                release_needed = False
                raise KVCRServiceError(response.message)
            grant_received = True
            spec = _grant_spec(response, pool_index, tier_config)
            if listener_fd is None:
                # Every pool has a Guard, and a Guard answers on the endpoint
                # this claimant named. A grant without it means the two sides
                # disagree about who serves this pool.
                raise KVCRGuardProtocolError(
                    "claim was granted without the endpoint it answers on"
                )
            try:
                effective_bytes, rows = _compute_pool_geometry(
                    spec.data_bytes, tier_config.row_stride
                )
            except ValueError as geometry_error:
                raise KVCRGuardProtocolError(
                    "invalid pool grant: mapping must contain the journal and at "
                    "least one complete KV row"
                ) from geometry_error
            attachment = KVCRPoolAttachment.attach(spec)
            return KVCRPoolHold(
                local_dram=LocalDramInfo(
                    attachment.data_address, effective_bytes, rows
                ),
                _attachment=attachment,
                _connection=connection,
                _control_listener_fd=listener_fd,
            )
        except BaseException as error:
            _abandon_claim(connection, attachment, listener_fd, release_needed)
            if not grant_received and isinstance(error, (OSError, EOFError)):
                raise KVCRSocketError(f"KVCR-Service claim failed: {error}") from error
            raise


def _abandon_claim(
    connection: FramedConnection,
    attachment: KVCRPoolAttachment | None,
    listener_fd: int | None,
    release_needed: bool,
) -> None:
    """Give back everything a claim took before it failed.

    The lease is only released once local access has actually stopped: telling
    the service the pool is free while this process can still reach its bytes
    is what would let the next claimant map them underneath us.
    """
    if listener_fd is not None:
        with contextlib.suppress(OSError):
            os.close(listener_fd)
    local_access_stopped = True
    if attachment is not None:
        try:
            attachment.close()
        except BaseException:
            # BaseException: even an interrupt mid-close must gate the release;
            # the claim error re-raises above.
            local_access_stopped = False
    if release_needed and local_access_stopped:
        # Best-effort during unwind: nothing raised here may mask the claim
        # error re-raised by the caller.
        with contextlib.suppress(BaseException):
            _send_release(connection)
    _close_quietly(connection)


def _claim_request(
    pool_index: int,
    row_stride: int,
    compatibility_digest: str,
    g3: G3Options | None,
    control_bind: tuple[str, int],
) -> _Claim:
    """Build the wire form of a claim, validating it before it is sent.

    msgspec validates on convert and not on __init__, so a bad stride, port or G3
    path is caught here rather than on the service's side of the socket.
    """
    return msgspec.convert(
        {
            "pool_index": pool_index,
            "compatibility_digest": compatibility_digest,
            "tier_config": {
                "row_stride": row_stride,
                "g3": (
                    None
                    if g3 is None
                    else {
                        "paths": [
                            str(path.expanduser().resolve()) for path in g3.paths
                        ],
                        "capacity_bytes_per_file": g3.capacity_bytes_per_file,
                        "backend": g3.backend,
                        "backend_options": dict(g3.backend_options),
                    }
                ),
            },
            "control_host": control_bind[0],
            "control_port": control_bind[1],
            "version": _PROTOCOL_VERSION,
        },
        type=_Claim,
    )


def _grant_spec(
    response: _Granted,
    requested_pool_index: int,
    requested_tier_config: _TierConfig,
) -> KVCRPoolSpec:
    """Take the grant apart, refusing one that answers a different request."""
    if response.pool_index != requested_pool_index:
        raise KVCRGuardProtocolError(
            "claim pool mismatch: "
            f"requested {requested_pool_index}, got {response.pool_index}"
        )
    if response.tier_config != requested_tier_config:
        raise KVCRGuardProtocolError("claim tier configuration mismatch")
    return response.spec


def _send_release(connection: FramedConnection) -> None:
    connection.send(_Release(_PROTOCOL_VERSION))
    response = connection.receive(_RELEASE_RESPONSE_DECODER)
    if isinstance(response, _Error):
        raise KVCRServiceError(response.message)


def _close_quietly(connection: FramedConnection) -> None:
    # Only called while an original error is unwinding; a close failure
    # (even an interrupt) must not mask it.
    with contextlib.suppress(BaseException):
        connection.close()
