# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control and framing channels used by KVCR."""

import logging
import os
import socket
import struct
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
        payload = _ENCODER.encode(message)
        if len(payload) > _MAX_FRAME_BYTES:
            raise KVCRMsgFramingError("frame is too large")
        self._connection.sendall(_FRAME_HEADER.pack(len(payload)) + payload)

    def receive(self, decoder: msgspec.msgpack.Decoder[_T]) -> _T:
        header = self._receive_exact(_FRAME_HEADER.size)
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

    ``send`` returns immediately when the ZMQ queue cannot accept data;
    KVCR operation deadlines handle retry and failure decisions.

    KVCR constructs the channel on main but initializes and uses its sockets
    only from the progress thread.
    """

    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        advertise_host: str,
    ) -> None:
        self._bind_host = bind_host
        self._bind_port = bind_port
        if bind_port <= 0:
            raise ValueError("KVCR control_port must be configured")
        self.endpoint = f"tcp://{advertise_host}:{bind_port}"
        self._ctx: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._outgoing: dict[str, zmq.Socket] = {}

    def initialize(self) -> None:
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PULL)
        self._socket.linger = 0
        self._socket.bind(f"tcp://{self._bind_host}:{self._bind_port}")

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
        for outgoing_socket in self._outgoing.values():
            outgoing_socket.close()
        self._outgoing.clear()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
