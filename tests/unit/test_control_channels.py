# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the progress-owned ZMQ peer control channel."""

import os
import socket
import time
from array import array
from collections.abc import Iterator
from contextlib import closing, contextmanager, suppress
from unittest.mock import Mock

import msgspec
import pytest
import zmq
from _kvcr_test_utils import free_port, listening_socket

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


@contextmanager
def _channel(
    port: int, advertise_host: str = "127.0.0.1"
) -> Iterator[ZmqPeerControlChannel]:
    with closing(ZmqPeerControlChannel("127.0.0.1", port, advertise_host)) as channel:
        yield channel


@contextmanager
def _push_socket(linger: int = 0) -> Iterator[zmq.Socket]:
    sender = zmq.Context.instance().socket(zmq.PUSH)
    sender.linger = linger
    with closing(sender):
        yield sender


@contextmanager
def _pipe() -> Iterator[tuple[int, int]]:
    """A pipe whose own ends are closed even if what was sent over them is not."""
    read_fd, write_fd = os.pipe()
    try:
        yield read_fd, write_fd
    finally:
        for file_descriptor in (read_fd, write_fd):
            with suppress(OSError):
                os.close(file_descriptor)


@contextmanager
def _recorded_closes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """Every descriptor os.close is asked for while this is active."""
    closed: list[int] = []
    real_close = os.close

    def close(file_descriptor: int) -> None:
        closed.append(file_descriptor)
        real_close(file_descriptor)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "close", close)
        yield closed


def _push_messages(endpoint: str, count: int) -> None:
    # Lingering, because the test reads these after the socket is gone.
    with _push_socket(linger=200) as sender:
        sender.connect(endpoint)
        for index in range(count):
            sender.send(msgspec.msgpack.encode({"seq": index}))


def _recv_until(channel: ZmqPeerControlChannel, count: int) -> list[bytes]:
    deadline = time.monotonic() + 1
    messages: list[bytes] = []
    while len(messages) < count and time.monotonic() < deadline:
        messages.extend(channel.recv())
        if len(messages) < count:
            time.sleep(0.001)
    return messages


@pytest.fixture
def pair() -> Iterator[Pair]:
    sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with sender, receiver:
        yield sender, receiver


@pytest.fixture
def channel() -> Iterator[ZmqPeerControlChannel]:
    with _channel(free_port()) as ready:
        ready.initialize()
        yield ready


def test_construction_does_not_claim_the_port_until_it_is_needed() -> None:
    """A channel constructs while its port is held; the conflict surfaces at use."""
    port = free_port()
    # A returning primary must construct its channel while a Guard holds the port.
    with (
        listening_socket(port) as holder,
        _channel(port) as channel,
    ):
        # A real conflict is still reported, just at the point of use.
        with pytest.raises(OSError):
            channel.initialize()

        holder.close()

        channel.initialize()


def test_messages_keep_their_boundaries(pair: Pair) -> None:
    sender, receiver = pair
    claim = _Message("claim", {"pool_index": 3})
    FramedConnection(sender).send(claim)
    FramedConnection(sender).send(_Message("ping"))
    incoming = FramedConnection(receiver)

    assert incoming.receive(_MESSAGE_DECODER) == claim
    assert incoming.receive(_MESSAGE_DECODER) == _Message("ping")


def test_a_frame_carries_at_most_one_descriptor(
    pair: Pair,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame carries one cloexec fd; refused deliveries close every received fd."""
    sender, receiver = pair
    receiver.settimeout(1)
    incoming = FramedConnection(receiver)

    # One descriptor accompanies its frame and arrives non-inheritable.
    received_fd: int | None = None
    with _pipe() as (read_fd, _write_fd):
        try:
            message = _Message("claim", {"pool_index": 3})
            FramedConnection(sender).send_with_fd(message, read_fd)

            received, received_fd = incoming.receive_with_fd(_MESSAGE_DECODER)

            assert received == message
            assert received_fd is not None
            assert os.get_inheritable(received_fd) is False
            assert os.fstat(received_fd).st_ino == os.fstat(read_fd).st_ino
        finally:
            if received_fd is not None:
                os.close(received_fd)

    # More than one descriptor on a frame is refused, and every one is closed.
    with _pipe() as (first, _first_peer), _pipe() as (second, _second_peer):
        payload = msgspec.msgpack.encode(_Message("claim"))
        frame = _FRAME_HEADER.pack(len(payload)) + payload
        sender.sendmsg(
            [frame],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", [first, second]))],
        )

        with _recorded_closes(monkeypatch) as closed:
            with pytest.raises(KVCRMsgFramingError, match="multiple"):
                incoming.receive_with_fd(_MESSAGE_DECODER)

        assert len(closed) == 2
        receiver.recv(len(payload))  # Drain the body of the refused frame.

    # A frame header that stops early is a truncation; its descriptor is closed.
    with _pipe() as (late_fd, _late_peer):
        sender.sendmsg(
            [_FRAME_HEADER.pack(64)[:2]],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", [late_fd]))],
        )
        sender.close()

        with _recorded_closes(monkeypatch) as closed:
            with pytest.raises(KVCRMsgFramingError, match="truncated"):
                incoming.receive_with_fd(_MESSAGE_DECODER)

        assert len(closed) == 1


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


def test_recv_is_bounded_per_progress_turn(channel: ZmqPeerControlChannel) -> None:
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
    with _channel(free_port()) as channel:
        channel._socket = fake_socket

        assert channel.recv() == []
        fake_socket.poll.assert_called_once_with(0)


def test_close_gives_up_every_socket_even_when_one_will_not_go() -> None:
    """Stopping at the first failure would leave the pool's address bound."""
    with listening_socket() as owner:
        port = int(owner.getsockname()[1])
        channel = ZmqPeerControlChannel("127.0.0.1", port, "127.0.0.1")
        channel.adopt_listener(os.dup(owner.fileno()))
        channel.initialize()
        listener = channel._listener
        pull = channel._socket
        assert listener is not None and pull is not None
        stubborn = Mock()
        stubborn.close.side_effect = OSError("outgoing will not close")
        channel._outgoing = {"tcp://peer": stubborn}

        with pytest.raises(OSError, match="will not close"):
            channel.close()

        # The address has to be free whatever the outgoing socket did.
        stubborn.close.assert_called_once_with()
        assert listener.fileno() == -1
        assert pull.closed
        # And a second close finds nothing left to do.
        channel.close()


def test_an_adopted_listener_advertises_its_port_and_outlives_the_primary() -> None:
    """An adopted listener advertises its port, serves the primary, then the Guard."""
    # The service owns the endpoint; the worker adopts it, a Guard duplicates it.
    with listening_socket() as owner:
        port = int(owner.getsockname()[1])
        guarded = ZmqPeerControlChannel.from_shared_listener(
            socket.socket(fileno=os.dup(owner.fileno()))
        )
        with (
            _channel(port, "advertised-host") as primary,
            closing(guarded) as guard,
            _push_socket() as sender,
        ):
            primary.adopt_listener(os.dup(owner.fileno()))
            # A primary needs a real advertised endpoint; a Guard reflects routes.
            assert primary.endpoint == f"tcp://advertised-host:{port}"
            # A whole instance, not one assembled by __new__.
            assert guard.control_bind_address() == ("127.0.0.1", port)
            owner.close()

            primary.initialize()
            with pytest.raises(RuntimeError):
                primary.adopt_listener(-1)
            sender.connect(f"tcp://127.0.0.1:{port}")
            sender.send(b"primary")
            assert _recv_until(primary, 1) == [b"primary"]

            primary.close()
            guard.initialize()
            deadline = time.monotonic() + 2
            received: list[bytes] = []
            while not received and time.monotonic() < deadline:
                sender.send(b"guard")
                time.sleep(0.01)
                received.extend(guard.recv())
            assert b"guard" in received
