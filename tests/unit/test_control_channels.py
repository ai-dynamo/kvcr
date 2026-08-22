# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the progress-owned ZMQ peer control channel."""

import socket
import time
from collections.abc import Iterator
from unittest.mock import Mock

import msgspec
import pytest
import zmq

from kvcr.control_channels import (
    _FRAME_HEADER,
    _MAX_FRAME_BYTES,
    FramedConnection,
    KVCRGuardProtocolError,
    KVCRMsgFramingError,
    ZmqPeerControlChannel,
)

Pair = tuple[socket.socket, socket.socket]


class _Message(msgspec.Struct, frozen=True):
    operation: str
    arguments: dict[str, int] | None = None


_MESSAGE_DECODER = msgspec.msgpack.Decoder(_Message)


@pytest.fixture
def pair() -> Iterator[Pair]:
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with sender, receiver:
        yield sender, receiver


def test_messages_keep_their_boundaries(pair: Pair) -> None:
    sender, receiver = pair
    claim = _Message("claim", {"pool_index": 3})
    FramedConnection(sender).send(claim)
    FramedConnection(sender).send(_Message("ping"))
    incoming = FramedConnection(receiver)

    assert incoming.receive(_MESSAGE_DECODER) == claim
    assert incoming.receive(_MESSAGE_DECODER) == _Message("ping")


def test_a_departed_peer_ends_the_stream(pair: Pair) -> None:
    sender, receiver = pair
    sender.close()

    with pytest.raises(EOFError):
        FramedConnection(receiver).receive(_MESSAGE_DECODER)


def test_a_frame_that_stops_early_is_a_truncation(pair: Pair) -> None:
    sender, receiver = pair
    sender.sendall(_FRAME_HEADER.pack(64) + b"not the whole 64")
    sender.close()

    with pytest.raises(KVCRMsgFramingError, match="truncated"):
        FramedConnection(receiver).receive(_MESSAGE_DECODER)


def test_an_oversized_outbound_frame_is_refused() -> None:
    connection = Mock()

    with pytest.raises(KVCRMsgFramingError, match="too large"):
        FramedConnection(connection).send({"payload": b"x" * _MAX_FRAME_BYTES})

    connection.sendall.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (msgspec.msgpack.encode([1, 2, 3]), KVCRGuardProtocolError),
        (b"\xc1", KVCRMsgFramingError),
    ],
)
def test_invalid_payload_is_refused_by_typed_decoder(
    pair: Pair,
    payload: bytes,
    expected_error: type[Exception],
) -> None:
    sender, receiver = pair
    sender.sendall(_FRAME_HEADER.pack(len(payload)) + payload)

    with pytest.raises(expected_error, match="invalid message payload"):
        FramedConnection(receiver).receive(_MESSAGE_DECODER)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _push_messages(endpoint: str, count: int) -> None:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.linger = 200
    sock.connect(endpoint)
    for index in range(count):
        sock.send(msgspec.msgpack.encode({"seq": index}))
    sock.close()


def _recv_until(channel: ZmqPeerControlChannel, count: int) -> list[bytes]:
    deadline = time.monotonic() + 1
    messages: list[bytes] = []
    while len(messages) < count and time.monotonic() < deadline:
        messages.extend(channel.recv())
        if len(messages) < count:
            time.sleep(0.001)
    return messages


@pytest.fixture()
def channel():
    ch = ZmqPeerControlChannel(
        bind_host="127.0.0.1",
        bind_port=_free_port(),
        advertise_host="127.0.0.1",
    )
    ch.initialize()
    yield ch
    ch.close()


def test_recv_is_bounded_per_progress_turn(channel) -> None:
    _push_messages(channel.endpoint, 65)
    time.sleep(0.02)

    first = channel.recv()
    second = _recv_until(channel, 1)

    assert len(first) == 64
    assert len(second) == 1
    assert {
        msgspec.msgpack.decode(message)["seq"] for message in first + second
    } == set(range(65))


def test_recv_checks_for_data_without_blocking() -> None:
    fake_socket = Mock()
    fake_socket.poll.return_value = False
    channel = ZmqPeerControlChannel(
        bind_host="127.0.0.1",
        bind_port=_free_port(),
        advertise_host="127.0.0.1",
    )
    channel._socket = fake_socket

    assert channel.recv() == []
    fake_socket.poll.assert_called_once_with(0)
