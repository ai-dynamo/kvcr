# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control and framing channels used by KVCR.

A pool's control endpoint outlives the worker holding it. The service binds
the port once and passes the listening socket itself, as a descriptor over the
claim connection, so the address is never reopened and no other process can
take it in between.
"""

import logging
import os
import socket
import struct
from array import array
from contextlib import ExitStack
from typing import TypeVar

import msgspec
import zmq

logger = logging.getLogger(__name__)

_FRAME_HEADER = struct.Struct("!I")
_MAX_FRAME_BYTES = 1 << 20
_TIMEOUT_SECONDS = 60.0
_MAX_RECV_BATCH = 64
_ENCODER = msgspec.msgpack.Encoder()
_T = TypeVar("_T")


class KVCRServiceError(RuntimeError):
    """KVCR-Service refused the request.

    A healthy service saying no. Callers must fail rather than silently
    substituting local memory.
    """


class KVCRSocketError(KVCRServiceError):
    """KVCR-Service could not be reached: connect, connection, or timeout."""


class KVCRMsgFramingError(KVCRServiceError):
    """A malformed length-prefixed message frame."""


class KVCRGuardProtocolError(KVCRServiceError):
    """A malformed or incompatible KVCR-Guard protocol message."""


class FramedConnection:
    """Exchange length-prefixed msgpack messages over one socket."""

    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection

    @classmethod
    def connect(
        cls,
        endpoint: str | os.PathLike[str],
    ) -> "FramedConnection":
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(_TIMEOUT_SECONDS)
            connection.connect(os.fspath(endpoint))
        except OSError as error:
            connection.close()
            raise KVCRSocketError(
                f"KVCR-Service is unreachable at {os.fspath(endpoint)}: {error}"
            ) from error
        return cls(connection)

    def close(self) -> None:
        self._connection.close()

    def send(self, message: object) -> None:
        self._connection.sendall(self._encode_frame(message))

    def send_with_fd(self, message: object, file_descriptor: int) -> None:
        """Send one frame carrying one descriptor on its first byte."""
        frame = self._encode_frame(message)
        sent = socket.send_fds(self._connection, [frame], [file_descriptor])
        self._connection.sendall(frame[sent:])

    @staticmethod
    def _encode_frame(message: object) -> bytes:
        payload = _ENCODER.encode(message)
        if len(payload) > _MAX_FRAME_BYTES:
            raise KVCRMsgFramingError("frame is too large")
        return _FRAME_HEADER.pack(len(payload)) + payload

    def receive(self, decoder: msgspec.msgpack.Decoder[_T]) -> _T:
        header = self._receive_exact(_FRAME_HEADER.size)
        return self._receive_after_header(header, decoder)

    def receive_with_fd(
        self, decoder: msgspec.msgpack.Decoder[_T]
    ) -> tuple[_T, int | None]:
        """Receive one frame and at most one descriptor attached to it."""
        # Not socket.recv_fds: it drops the flags it is given, so MSG_CMSG_CLOEXEC
        # would never reach recvmsg.
        header, ancillary, flags, _ = self._connection.recvmsg(
            _FRAME_HEADER.size,
            socket.CMSG_SPACE(array("i").itemsize * 2),
            socket.MSG_CMSG_CLOEXEC,
        )
        if not header:
            raise EOFError

        descriptors = array("i")
        try:
            unexpected_ancillary = False
            for level, kind, data in ancillary:
                if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                    unexpected_ancillary = True
                    continue
                size = len(data) - len(data) % descriptors.itemsize
                descriptors.frombytes(data[:size])
            if flags & socket.MSG_CTRUNC:
                raise KVCRMsgFramingError("truncated ancillary data")
            if unexpected_ancillary:
                raise KVCRMsgFramingError("unexpected ancillary data")
            if len(descriptors) > 1:
                raise KVCRMsgFramingError("multiple file descriptors received")
            if len(header) < _FRAME_HEADER.size:
                try:
                    header += self._receive_exact(_FRAME_HEADER.size - len(header))
                except EOFError as error:
                    raise KVCRMsgFramingError("truncated frame") from error
            message = self._receive_after_header(header, decoder)
            return message, descriptors[0] if descriptors else None
        except BaseException:
            for file_descriptor in descriptors:
                os.close(file_descriptor)
            raise

    def _receive_after_header(
        self, header: bytes, decoder: msgspec.msgpack.Decoder[_T]
    ) -> _T:
        (length,) = _FRAME_HEADER.unpack(header)
        if length == 0 or length > _MAX_FRAME_BYTES:
            raise KVCRMsgFramingError(f"invalid frame length: {length}")
        payload = self._receive_exact(length)
        try:
            return decoder.decode(payload)
        except msgspec.ValidationError as error:
            raise KVCRGuardProtocolError(f"invalid message payload: {error}") from error
        except msgspec.DecodeError as error:
            raise KVCRMsgFramingError(f"invalid message payload: {error}") from error

    def _receive_exact(self, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self._connection.recv(length - len(chunks))
            if not chunk:
                if not chunks:
                    raise EOFError
                raise KVCRMsgFramingError("truncated frame")
            chunks.extend(chunk)
        return bytes(chunks)


class ZmqPeerControlChannel:
    """Nonblocking peer channel.

    ``send`` returns immediately when the ZMQ queue cannot accept data; KVCR
    operation deadlines handle retry and failure. Constructed on main, but its
    sockets are only touched from the progress thread.
    """

    endpoint: str | None
    _bind_endpoint: str
    _listener: socket.socket | None
    _ctx: zmq.Context | None
    _socket: zmq.Socket | None
    _outgoing: dict[str, zmq.Socket]

    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        advertise_host: str,
    ) -> None:
        if bind_port <= 0:
            raise ValueError("KVCR control_port must be configured")
        self._bind_host = bind_host
        self._bind_port = bind_port
        self._advertise_host = advertise_host
        self.endpoint = f"tcp://{advertise_host}:{bind_port}"
        self._bind_endpoint = f"tcp://{bind_host}:{bind_port}"
        self._listener = None
        self._ctx = None
        self._socket = None
        self._outgoing = {}

    @classmethod
    def from_shared_listener(cls, listener: socket.socket) -> "ZmqPeerControlChannel":
        """Take ownership of a pre-bound TCP listener received by the service."""
        if listener.family != socket.AF_INET:
            raise ValueError("KVCR control listener must use TCP")
        if listener.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
            raise ValueError("KVCR control listener must be a stream socket")
        if listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1:
            raise ValueError("KVCR control listener must already be listening")
        host, port = listener.getsockname()[:2]
        channel = cls(host, int(port), host)
        channel.adopt_listener(listener.detach())
        # The service cannot reconstruct the primary's advertised host; the
        # Guard replies using routes reflected in incoming requests.
        channel.endpoint = None
        return channel

    def control_bind_address(self) -> tuple[str, int]:
        return self._bind_host, self._bind_port

    def adopt_listener(self, listener_fd: int) -> None:
        """Take over a listener bound elsewhere instead of binding one.

        The service owns a pool's control endpoint across the death of the
        worker holding it, so a worker adopts that listener rather than
        competing for the port the service is about to hand it.
        """
        if self._listener is not None:
            raise RuntimeError("KVCR control channel already owns a listener")
        listener = socket.socket(fileno=listener_fd)
        try:
            host, port = listener.getsockname()[:2]
        except BaseException:
            # Detached, not closed: the hold owns this descriptor until the
            # adoption returns, and closes it on release; closing here would
            # have the release double-close it.
            listener.detach()
            raise
        self._listener = listener
        self._bind_endpoint = f"tcp://{host}:{int(port)}"
        self.endpoint = f"tcp://{self._advertise_host}:{int(port)}"

    def initialize(self) -> None:
        if self._listener is None:
            self._listener = socket.create_server((self._bind_host, self._bind_port))
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PULL)
        self._socket.linger = 0
        adopted_fd = os.dup(self._listener.fileno())
        try:
            self._socket.setsockopt(zmq.USE_FD, adopted_fd)
            self._socket.bind(self._bind_endpoint)
        except BaseException:
            # libzmq takes the descriptor at bind, so one that failed to bind
            # leaves it ours -- and it holds the pool's control address.
            os.close(adopted_fd)
            self._socket.close()
            self._socket = None
            raise

    def send(self, endpoint: str, message: bytes) -> bool:
        if self._ctx is None:
            raise RuntimeError("KVCR control channel is not initialized")
        socket = self._outgoing.get(endpoint)
        if socket is None:
            socket = self._ctx.socket(zmq.PUSH)
            socket.linger = 0
            try:
                socket.connect(endpoint)
            except zmq.ZMQError:
                logger.warning("KVCR control connect failed to %s", endpoint)
                socket.close()
                return False
            self._outgoing[endpoint] = socket
        try:
            socket.send(message, flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.warning("KVCR control send queue is full for %s", endpoint)
            socket.close()
            self._outgoing.pop(endpoint, None)
            return False
        except zmq.ZMQError:
            logger.warning("KVCR control send failed to %s", endpoint)
            socket.close()
            self._outgoing.pop(endpoint, None)
            return False
        return True

    def recv(self) -> list[bytes]:
        socket = self._socket
        if socket is None:
            raise RuntimeError("KVCR control channel is not initialized")
        messages: list[bytes] = []
        try:
            if not socket.poll(0):
                return messages
            while len(messages) < _MAX_RECV_BATCH:
                messages.append(socket.recv(flags=zmq.NOBLOCK))
        except zmq.Again:
            return messages
        except zmq.ZMQError:
            logger.warning("KVCR control recv failed", exc_info=True)
        return messages

    def close(self) -> None:
        """Give up every socket, whatever any one of them does about it.

        A first-failure stop leaves the pool's control address bound, unclaimable.
        """
        with ExitStack() as sockets:
            for target in (*self._outgoing.values(), self._socket, self._listener):
                if target is not None:
                    sockets.callback(target.close)
            # Dropped before any close runs: a raising close is not retried.
            self._outgoing = {}
            self._socket = None
            self._listener = None
