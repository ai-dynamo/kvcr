# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-process Guard promotion through the journal and peer G2 path."""

import ctypes
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeNixlAgent,
    FakePrimaryPinning,
    _has_outstanding_operations,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _use_nixl_agent,
    _wait_until,
    free_port,
)

from kvcr import KVCR, KVCRBindings
from kvcr import progress as kvcr_progress
from kvcr.config import (
    FrameworkDramInput,
    G3Options,
    KVCRBackendConfigs,
    KVCRConfig,
    KVCRGuardConfig,
    RemoteFWDramOptions,
)
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.guard import _Guard
from kvcr.kvcr_service import _KVCRService
from kvcr.types import BlockKey, CacheTier, QueryStatus

_TIMEOUT_SECONDS = 5
# A real NIXL agent and its UCX backend dominate a child's startup.
_REAL_NIXL_TIMEOUT_SECONDS = 60
_DIGEST = "01" * 32


class _FileBackedNixlAgent(FakeNixlAgent):
    def __init__(self) -> None:
        super().__init__(b"guard-md")
        self.name = b"target"
        self.block_remote_writes = False
        self.blocked_remote_writes = 0
        self.backends: dict[str, dict[str, str]] = {}

    def add_remote_agent(self, metadata: bytes) -> bytes:
        self.remote_agents.append(metadata)
        return b"target"

    def get_plugin_list(self) -> list[str]:
        return ["MOCK"]

    def create_backend(self, backend: str, options: dict[str, str]) -> None:
        self.backends[backend] = dict(options)

    def get_backend_params(self, backend: str) -> dict[str, str]:
        return self.backends[backend]

    def register_memory(self, descs, mem_type="DRAM", backends=None):
        return super().register_memory(descs, mem_type)

    def deregister_memory(self, handle, backends=None):
        return super().deregister_memory(handle)

    def transfer(self, handle):
        # The base class already records per-xfer backends; none means DRAM.
        if not self.xfer_backends[handle - 1]:
            if self.block_remote_writes:
                self.transfers.append(handle)
                self.blocked_remote_writes += 1
                return "PROC"
            return super().transfer(handle)
        self.transfers.append(handle)
        operation, local_descs, local_indices, file_descs, _, _ = self.xfers[handle - 1]
        for index in local_indices:
            address, memory_bytes, _ = local_descs[index]
            offset, file_bytes, direct_fd = file_descs[index]
            byte_count = min(memory_bytes, file_bytes)
            fd = os.open(f"/proc/self/fd/{direct_fd}", os.O_RDWR | os.O_CLOEXEC)
            try:
                if operation == "WRITE":
                    os.pwrite(fd, ctypes.string_at(address, byte_count), offset)
                else:
                    data = os.pread(fd, byte_count, offset)
                    ctypes.memmove(address, data, len(data))
            finally:
                os.close(fd)
        return "DONE"


def _make_kvcr(
    socket_path: str,
    g3_path: str,
    control_port: int | str,
    agent_name: str,
    *,
    agent: FakeNixlAgent | None = None,
    framework: "ctypes.Array | None" = None,
) -> KVCR:
    """A claiming KVCR: a fake agent gets a MOCK G3, a real one POSIX plus
    NIXL-registered framework memory every descriptor it hands KVCR points into."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    pinning = FakePrimaryPinning()
    with _use_nixl_agent(agent) if agent is not None else nullcontext():
        return KVCR(
            KVCRConfig(
                nixl_agent_name=agent_name,
                nixl_listen_port=0,
                inventory_report_interval_ms=0,
            ),
            KVCRBindings(
                pinning.request_pin,
                pinning.poll_pin_results,
                pinning.release_pin,
                framework_control=ZmqPeerControlChannel(
                    "127.0.0.1", int(control_port), "127.0.0.1"
                ),
            ),
            KVCRBackendConfigs(
                framework_dram=(
                    FrameworkDramInput(ctypes.addressof(framework), len(framework))
                    if framework is not None
                    else None
                ),
                g3=G3Options(
                    paths=(Path(g3_path),),
                    capacity_bytes_per_file=(
                        page_size if agent is not None else page_size * 2
                    ),
                    backend="MOCK" if agent is not None else "POSIX",
                ),
                remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
            ),
            KVCRGuardConfig(
                kvcr_service_socket_path=socket_path,
                pool_index=0,
                row_stride=page_size,
                compatibility_digest=_DIGEST,
            ),
        )


def _deposit_two_blocks(kvcr: KVCR, source: int, block_bytes: int) -> None:
    """Fill the pool, leaving resident-b in G2 and resident-a spilled to G3."""
    for key, payload in (
        (BlockKey(b"resident-a"), b"A" * block_bytes),
        (BlockKey(b"resident-b"), b"B" * block_bytes),
    ):
        ctypes.memmove(source, payload, block_bytes)
        operation = kvcr.deposit({key: _mem_descriptor(source, block_bytes)})
        assert dict(_poll_until(kvcr, bool))[operation][key].success, key
    assert kvcr.query((BlockKey(b"resident-a"),)) == [
        (QueryStatus.FETCHABLE, CacheTier.G3)
    ]


def _primary_child(
    socket_path: str, g3_path: str, control_port: str, mode: str
) -> None:
    """Claim the pool, deposit unless idle, then hold it -- stalled mid-write
    when asked -- until killed."""
    slot_bytes = os.sysconf("SC_PAGE_SIZE")
    agent = _FileBackedNixlAgent()
    agent.state = "DONE"
    kvcr = _make_kvcr(socket_path, g3_path, control_port, mode, agent=agent)
    if mode != "idle":
        source = ctypes.create_string_buffer(slot_bytes)
        _deposit_two_blocks(kvcr, ctypes.addressof(source), slot_bytes)
    if mode == "stall":
        agent.block_remote_writes = True
        agent.state = "PROC"
        print("stable", flush=True)
        while not agent.blocked_remote_writes:
            kvcr.poll_completed()
            time.sleep(0.001)
        print("in-flight", flush=True)
    else:
        print("ready", flush=True)
    time.sleep(60)


def _stale_peer_child(control_port: str, probe_port: str) -> None:
    """A dead primary's peer: it sends into the pool's endpoint and must get
    a terminal refusal back, not silence until its operation deadline."""
    import zmq

    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVTIMEO, int(_TIMEOUT_SECONDS * 1000))
    pull.bind(f"tcp://127.0.0.1:{probe_port}")
    push = context.socket(zmq.PUSH)
    push.connect(f"tcp://127.0.0.1:{control_port}")
    push.send(
        msgspec.msgpack.encode(
            {
                "type": "start_write",
                "op_handle": 7,
                "remaining_timeout_ms": 1000,
                "keys": [],
                "dst_descriptors": [],
                "sender_control_endpoint": f"tcp://127.0.0.1:{probe_port}",
                "source_control_endpoint": f"tcp://127.0.0.1:{control_port}",
            }
        )
    )
    while True:
        decoded = msgspec.msgpack.decode(pull.recv())
        if decoded.get("type") == "write_refused" and decoded["op_handle"] == 7:
            print("refused", flush=True)
            return


@pytest.fixture
def live_service(
    tmp_path: Path,
) -> Iterator[tuple[_KVCRService, Callable[..., subprocess.Popen[str]]]]:
    """A one-pool service on its own thread; children it spawns die with it."""
    pool_dir = tmp_path / "pools"
    pool_dir.mkdir()
    service = _KVCRService(
        tmp_path / "service.sock",
        pool_dir,
        pool_count=1,
        pool_size_bytes=8192 + os.sysconf("SC_PAGE_SIZE"),
        journal_bytes=8192,
        compatibility_digest=_DIGEST,
    )
    server_thread = threading.Thread(target=service.serve_forever)
    server_thread.start()
    children: list[subprocess.Popen[str]] = []

    def spawn(child_function: str, *args: object) -> subprocess.Popen[str]:
        bootstrap = (
            "import sys; "
            "sys.path.insert(0, sys.argv[1]); "
            f"from test_guard_integration import {child_function}; "
            f"{child_function}(*sys.argv[2:])"
        )
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                bootstrap,
                str(Path(__file__).parent),
                *(str(arg) for arg in args),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        return child

    yield service, spawn
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=_TIMEOUT_SECONDS)
        for stream in (child.stdin, child.stdout, child.stderr):
            if stream is not None:
                stream.close()
    service.shutdown()
    server_thread.join(timeout=_TIMEOUT_SECONDS)
    service.close()
    assert not server_thread.is_alive()


_RAN_BEFORE_REAL_NIXL: list[str] = []


@pytest.fixture(autouse=True)
def _real_nixl_runs_first(request: pytest.FixtureRequest) -> None:
    # Enforced, not just asked for: this module's real-NIXL scenario builds
    # real NIXL agents in this process, and real createXferReq starts failing
    # with NIXL_ERR_INVALID_PARAM when fake-agent/ZMQ scenarios have run here
    # first. Pre-existing sensitivity, reproduced without any of the
    # surrounding tests' recent changes; worth its own investigation. A
    # reordering (xdist, -p, a new test added above) fails loudly here instead
    # of as an inscrutable NIXL error.
    if "real_nixl" in request.node.name and _RAN_BEFORE_REAL_NIXL:
        pytest.fail(
            f"{request.node.name} must run first in this module; "
            f"{_RAN_BEFORE_REAL_NIXL[0]} already ran in this process"
        )
    _RAN_BEFORE_REAL_NIXL.append(request.node.name)


def test_a_promoted_guard_serves_real_nixl_transfers(
    tmp_path: Path,
    live_service: tuple[_KVCRService, Callable[..., subprocess.Popen[str]]],
) -> None:
    """With nothing faked, a promoted Guard serves a real UCX read then stands down."""
    # Not a decorator: children import this module, and NIXL logs to their stdout.
    if not _real_nixl_available():
        pytest.skip("no runnable NIXL agent on this machine")
    page_size = os.sysconf("SC_PAGE_SIZE")
    second_payload = b"B" * page_size
    g3_path = tmp_path / "g3.data"
    control_port = free_port()
    service, spawn = live_service

    primary = spawn(
        "_real_nixl_primary_child", service.socket_path, g3_path, control_port
    )
    _await_marker(primary, "ready", _REAL_NIXL_TIMEOUT_SECONDS)
    guard = service._registry._pools[0]

    primary.kill()
    primary.wait(timeout=_TIMEOUT_SECONDS)
    # Promotion builds a real agent under a new name over the same pool.
    _wait_until(lambda: guard._serving, timeout=_REAL_NIXL_TIMEOUT_SECONDS)
    assert set(guard._core._block_record_map) == {BlockKey(b"resident-b")}

    # A real UCX read through the Guard: the agent did not exist at write time.
    source_endpoint = f"tcp://127.0.0.1:{control_port}"
    target_memory = ctypes.create_string_buffer(page_size)
    target_pinning = FakePrimaryPinning()
    target = KVCR(
        KVCRConfig(
            nixl_agent_name="real-target",
            nixl_listen_port=0,
            inventory_report_interval_ms=0,
            operation_timeout_ms=_REAL_NIXL_TIMEOUT_SECONDS * 1000,
        ),
        KVCRBindings(
            target_pinning.request_pin,
            target_pinning.poll_pin_results,
            target_pinning.release_pin,
            framework_control=ZmqPeerControlChannel(
                "127.0.0.1", free_port(), "127.0.0.1"
            ),
        ),
        KVCRBackendConfigs(
            framework_dram=FrameworkDramInput(
                ctypes.addressof(target_memory), len(target_memory)
            ),
            remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
        ),
    )
    try:
        served_key = BlockKey(b"resident-b")
        target.submit_hint((served_key,), src=source_endpoint, request_id="from-guard")
        operation = target.deliver(
            {served_key: _mem_descriptor(ctypes.addressof(target_memory), page_size)},
            request_id="from-guard",
        )
        deadline = time.monotonic() + _REAL_NIXL_TIMEOUT_SECONDS
        results: dict = {}
        while time.monotonic() < deadline and not results:
            results = dict(target.poll_completed())
            time.sleep(0.01)
        assert results, "the Guard never answered the target"
        assert results[operation][served_key].success
        assert (
            ctypes.string_at(ctypes.addressof(target_memory), page_size)
            == second_payload
        )
        # _serving stands for "answering peers": a served read must not end it.
        assert guard._serving is True
    finally:
        target.close()

    # Only then does a replacement take the pool back and stand the Guard down.
    # Inline, not a child: adoption only claims and reads, and this process
    # already runs real agents beside the Guard's own.
    framework = ctypes.create_string_buffer(page_size * 2)
    replacement = _make_kvcr(
        str(service.socket_path),
        str(g3_path),
        control_port,
        "real-replacement",
        framework=framework,
    )
    try:
        destination = ctypes.addressof(framework) + page_size
        for key, payload in (
            (BlockKey(b"resident-a"), b"A" * page_size),
            (BlockKey(b"resident-b"), second_payload),
        ):
            ctypes.memset(destination, 0, len(payload))
            operation = replacement.deliver(
                {key: _mem_descriptor(destination, len(payload))}
            )
            result = dict(
                _poll_until(replacement, bool, timeout=_REAL_NIXL_TIMEOUT_SECONDS)
            )[operation][key]
            assert result.success, key
            assert ctypes.string_at(destination, len(payload)) == payload, key
        assert guard._serving is False
    finally:
        replacement.close()


@pytest.mark.parametrize("recovery", ["kept", "given-up"])
def test_request_timeout_during_promotion_then_retry_uses_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery: str,
    live_service: tuple[_KVCRService, Callable[..., subprocess.Popen[str]]],
) -> None:
    """A retry after promotion is served warm or refused cold, never left hanging."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    second_payload = b"B" * page_size
    g3_path = tmp_path / "g3.data"
    primary_port = free_port()
    source_endpoint = f"tcp://127.0.0.1:{primary_port}"
    guard_agent = _FileBackedNixlAgent()
    promotion_started = threading.Event()
    continue_promotion = threading.Event()
    promote = _Guard._promote

    promoted_with: list[int] = []

    def pause_promotion(guard: _Guard) -> None:
        promotion_started.set()
        assert continue_promotion.wait(timeout=_TIMEOUT_SECONDS)
        if recovery == "given-up":
            # What a ring the primary outran leaves behind.
            guard._mirror = None
        promote(guard)
        assert guard._core is not None
        promoted_with.append(len(guard._core._block_record_map))

    monkeypatch.setattr(_Guard, "_promote", pause_promotion)
    monkeypatch.setattr(kvcr_progress, "nixl_agent_config", lambda **kwargs: kwargs)
    monkeypatch.setattr(kvcr_progress, "nixl_agent", lambda _name, _config: guard_agent)

    service, spawn = live_service
    target = None
    try:
        # The stalled write must die with its process: only a SIGKILL mid-flight
        # leaves the journal the way a crashed primary would.
        child = spawn(
            "_primary_child", service.socket_path, g3_path, primary_port, "stall"
        )
        _await_marker(child, "stable")

        target_control = ZmqPeerControlChannel("127.0.0.1", free_port(), "127.0.0.1")
        target_agent = FakeNixlAgent(b"target-md")
        target_memory = ctypes.create_string_buffer(3 * page_size)
        target = _new_kvcr(
            target_agent,
            FakePrimaryPinning(),
            target_control,
            KVCRConfig(nixl_agent_name="target", operation_timeout_ms=5000),
            remote_options=RemoteFWDramOptions(eager_ctrl_connect=False),
            framework_dram=FrameworkDramInput(
                ctypes.addressof(target_memory), len(target_memory)
            ),
        )
        now = [0.0]
        target._core._clock = lambda: now[0]
        # The G2 block: a Guard serves what the pool holds and opens no G3.
        key = BlockKey(b"resident-b")
        stalled_destination = (ctypes.c_char * page_size).from_buffer(target_memory)
        target.submit_hint((key,), src=source_endpoint, request_id="stalled")
        target.deliver(
            {key: _mem_descriptor(ctypes.addressof(stalled_destination), page_size)},
            request_id="stalled",
        )
        _await_marker(child, "in-flight")
        _wait_until(
            lambda: (
                source_endpoint in target._core._remote_fw_dram._metadata_acked_sources
            ),
            timeout=2,
        )

        child.kill()
        child.wait(timeout=_TIMEOUT_SECONDS)
        assert promotion_started.wait(timeout=_TIMEOUT_SECONDS)
        guard = service._registry._pools[0]

        now[0] = 6.0
        _wait_until(
            lambda: (
                source_endpoint
                not in target._core._remote_fw_dram._metadata_acked_sources
            ),
            timeout=2,
        )
        assert list(target.poll_completed()) == []
        assert _has_outstanding_operations(target)

        continue_promotion.set()
        _wait_until(lambda: guard._serving, timeout=2)

        # A real core either way, answering on the endpoint it inherited.
        assert guard._core is not None
        destination = (ctypes.c_char * page_size).from_buffer(target_memory, page_size)
        target.submit_hint((key,), src=source_endpoint, request_id="retry")
        operation = target.deliver(
            {key: _mem_descriptor(ctypes.addressof(destination), len(destination))},
            request_id="retry",
        )

        if recovery == "given-up":
            # Serving nothing, so the request cannot be filled. It still has to
            # end: the Guard reports the failure rather than going quiet, which
            # is what the peer used to wait forever on.
            assert promoted_with == [0]
            _wait_until(lambda: guard_agent.sent_notifs, timeout=5)
            _, notification = guard_agent.sent_notifs[-1]
            target_agent.notifs[guard._core.nixl_agent_name] = [notification]
            completed = _poll_until(target, bool, timeout=5)
            assert completed[0][0] == operation
            assert not completed[0][1][key].success
            return

        _wait_until(lambda: bytes(destination) == second_payload, timeout=2)
        guard_agent.state = "DONE"
        _wait_until(lambda: guard_agent.released_xfers == [1], timeout=2)
        target_agent.notifs[guard._core.nixl_agent_name] = [guard_agent.xfers[0][5]]
        completed = _poll_until(target, bool, timeout=2)
        assert completed[0][0] == operation
        assert completed[0][1][key].success
    finally:
        continue_promotion.set()
        if target is not None:
            target.close()


def test_replacement_primary_takes_the_cache_back_from_a_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_service: tuple[_KVCRService, Callable[..., subprocess.Popen[str]]],
) -> None:
    """A Guard serves the dead primary's G2 block until a replacement adopts it."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    g3_path = tmp_path / "g3.data"
    control_port = free_port()
    guard_agent = _FileBackedNixlAgent()
    guard_agent.state = "DONE"
    monkeypatch.setattr(kvcr_progress, "nixl_agent_config", lambda **kwargs: kwargs)
    monkeypatch.setattr(kvcr_progress, "nixl_agent", lambda _name, _config: guard_agent)
    service, spawn = live_service

    # Generation zero: a primary that stored nothing dies. An empty Guard
    # must still serve -- one that declined would leave the dead primary's
    # peers sending into an endpoint nobody reads, stalling them until their
    # operation deadline instead of failing them now.
    idle = spawn("_primary_child", service.socket_path, g3_path, control_port, "idle")
    _await_marker(idle, "ready")
    first_guard = service._registry._pools[0]
    idle.kill()
    idle.wait(timeout=_TIMEOUT_SECONDS)
    _wait_until(lambda: first_guard._serving, timeout=_TIMEOUT_SECONDS)
    assert first_guard._core._block_record_map == {}

    # A stale peer's request gets a terminal refusal, not silence. In its own
    # process, as a real peer is -- and because a ZMQ probe in this process
    # destabilizes the real-NIXL test that follows.
    peer = spawn("_stale_peer_child", control_port, free_port())
    _await_marker(peer, "refused")
    peer.wait(timeout=_TIMEOUT_SECONDS)

    primary = spawn(
        "_primary_child", service.socket_path, g3_path, control_port, "held"
    )
    _await_marker(primary, "ready")

    primary.kill()
    primary.wait(timeout=_TIMEOUT_SECONDS)
    _wait_until(lambda: first_guard._serving, timeout=_TIMEOUT_SECONDS)
    # The G2 half is what a Guard serves; resident-a is spilled to G3.
    assert set(first_guard._core._block_record_map) == {BlockKey(b"resident-b")}

    # Inline, not a child: adoption only claims and reads, and the service's
    # pidfd on our own live pid never fires.
    replacement_agent = _FileBackedNixlAgent()
    replacement_agent.state = "DONE"
    replacement = _make_kvcr(
        str(service.socket_path),
        str(g3_path),
        control_port,
        "replacement",
        agent=replacement_agent,
    )
    try:
        for key, payload in (
            (BlockKey(b"resident-a"), b"A" * page_size),
            (BlockKey(b"resident-b"), b"B" * page_size),
        ):
            destination = ctypes.create_string_buffer(len(payload))
            operation = replacement.deliver(
                {key: _mem_descriptor(ctypes.addressof(destination), len(payload))}
            )
            result = dict(_poll_until(replacement, bool))[operation][key]
            assert result.success, key
            assert destination.raw == payload, key
    finally:
        replacement.close()


def _marker_lines(child: subprocess.Popen[str]) -> "queue.Queue[str]":
    """One drain thread per child, however many markers are awaited.

    A thread, not select(): TextIOWrapper may already hold the next line
    buffered. One per child, not per wait: a second drain thread would race
    the first for the stream and eat the marker it was started to find.
    """
    lines = getattr(child, "_marker_lines", None)
    if lines is None:
        assert child.stdout is not None
        lines = child._marker_lines = queue.Queue()
        reader = threading.Thread(target=_drain_lines, args=(child.stdout, lines))
        reader.daemon = True
        reader.start()
    return lines


def _await_marker(
    child: subprocess.Popen[str], marker: str, timeout: float = _TIMEOUT_SECONDS
) -> None:
    """Wait for a line the child meant to send, ignoring what NIXL logs."""
    lines = _marker_lines(child)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            child.kill()
            child.wait(timeout=_TIMEOUT_SECONDS)
            assert child.stderr is not None
            pytest.fail(f"child did not report {marker!r}: {child.stderr.read()}")
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            continue
        if not line:
            assert child.stderr is not None
            pytest.fail(f"child exited before {marker!r}: {child.stderr.read()}")
        if line.strip() == marker:
            return


def _drain_lines(stream, lines: "queue.Queue[str]") -> None:
    """Move whole lines off the pipe, ending with the empty string at EOF."""
    for line in stream:
        lines.put(line)
    lines.put("")


def _real_nixl_available() -> bool:
    """Whether this machine can actually run a NIXL agent, not just import one."""
    try:
        import nixl._api as api
    except ImportError:
        return False
    try:
        api.nixl_agent(
            "kvcr-availability-probe", api.nixl_agent_config(backends=["UCX"])
        )
    except (ImportError, RuntimeError, OSError):
        # Only the shapes an absent backend takes; anything else is a real failure.
        return False
    return True


def _real_nixl_primary_child(socket_path: str, g3_path: str, control_port: str) -> None:
    """Fill the pool through a real agent, then hold the claim until killed."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    framework = ctypes.create_string_buffer(page_size * 2)
    kvcr = _make_kvcr(
        socket_path, g3_path, control_port, "real-primary", framework=framework
    )
    _deposit_two_blocks(kvcr, ctypes.addressof(framework), page_size)
    print("ready", flush=True)
    time.sleep(60)
