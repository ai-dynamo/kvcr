# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-process Guard promotion through the journal and peer G2 path."""

import ctypes
import os
import queue
import select
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from _kvcr_test_utils import (
    FakeNixlAgent,
    FakePrimaryPinning,
    _has_outstanding_operations,
    _mem_descriptor,
    _new_kvcr,
    _poll_until,
    _wait_until,
    free_port,
)

from kvcr import KVCR, KVCRBindings
from kvcr import progress as kvcr_progress
from kvcr.config import (
    FrameworkDramInput,
    KVCRBackendConfigs,
    KVCRConfig,
    RemoteFWDramOptions,
)
from kvcr.control_channels import ZmqPeerControlChannel
from kvcr.guard import _Guard
from kvcr.kvcr_service import _KVCRService
from kvcr.types import BlockKey

_TIMEOUT_SECONDS = 5
# A real NIXL agent and its UCX backend dominate a child's startup.
_REAL_NIXL_TIMEOUT_SECONDS = 60
_DIGEST = "01" * 32


def _claiming_child(
    socket_path: Path,
    g3_path: Path,
    slot_bytes: int,
    control_port: int,
    agent_name: str,
) -> str:
    """The source every child shares: claim a service pool and serve from it."""
    test_utils = str(Path(__file__).parent)
    return textwrap.dedent(
        f"""
        import ctypes
        import sys
        import time
        from pathlib import Path

        sys.path.insert(0, {test_utils!r})
        from test_guard_integration import _FileBackedNixlAgent
        from _kvcr_test_utils import (
            FakePrimaryPinning,
            _mem_descriptor,
            _poll_until,
            _use_nixl_agent,
        )
        from kvcr import KVCR, KVCRBindings
        from kvcr.config import (
            G3Options,
            KVCRBackendConfigs,
            KVCRConfig,
            KVCRGuardConfig,
            RemoteFWDramOptions,
        )
        from kvcr.control_channels import ZmqPeerControlChannel
        from kvcr.types import BlockKey, CacheTier, QueryStatus

        first_key = BlockKey(b"resident-a")
        second_key = BlockKey(b"resident-b")
        agent = _FileBackedNixlAgent()
        agent.state = "DONE"
        pinning = FakePrimaryPinning()
        with _use_nixl_agent(agent):
            kvcr = KVCR(
                KVCRConfig(
                    nixl_agent_name={agent_name!r},
                    nixl_listen_port=0,
                    inventory_report_interval_ms=0,
                ),
                KVCRBindings(
                    pinning.request_pin,
                    pinning.poll_pin_results,
                    pinning.release_pin,
                    framework_control=ZmqPeerControlChannel(
                        "127.0.0.1", {control_port}, "127.0.0.1"
                    ),
                ),
                KVCRBackendConfigs(
                    g3=G3Options(
                        paths=(Path({str(g3_path)!r}),),
                        capacity_bytes_per_file={slot_bytes},
                        backend="MOCK",
                    ),
                    remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
                ),
                KVCRGuardConfig(
                    kvcr_service_socket_path={str(socket_path)!r},
                    pool_index=0,
                    row_stride={slot_bytes},
                    compatibility_digest={_DIGEST!r},
                ),
            )
        """
    )


def _deposit_source(first_payload: bytes, second_payload: bytes) -> str:
    """Fill the pool, leaving one block in G2 and one spilled to G3."""
    return textwrap.dedent(
        f"""
        for key, payload in (
            (first_key, {first_payload!r}),
            (second_key, {second_payload!r}),
        ):
            source = ctypes.create_string_buffer({len(first_payload)})
            source.raw = payload
            operation = kvcr.deposit(
                {{key: _mem_descriptor(ctypes.addressof(source), len(payload))}}
            )
            assert dict(_poll_until(kvcr, bool))[operation][key].success
        assert kvcr.query((first_key,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
        """
    )


def _primary_program(
    socket_path: Path,
    g3_path: Path,
    first_payload: bytes,
    second_payload: bytes,
    control_port: int,
) -> str:
    """A primary that fills the pool and then stalls a write mid-flight."""
    return (
        _claiming_child(
            socket_path, g3_path, len(first_payload), control_port, "primary"
        )
        + textwrap.dedent(
            r"""
        print("claimed", flush=True)
        assert sys.stdin.readline() == "deposit\n"
        """
        )
        + _deposit_source(first_payload, second_payload)
        + textwrap.dedent(
            """
        agent.block_remote_writes = True
        agent.state = "PROC"
        print("stable", flush=True)
        while not agent.blocked_remote_writes:
            kvcr.poll_completed()
            time.sleep(0.001)
        print("in-flight", flush=True)
        time.sleep(60)
        """
        )
    )


def _expect_child_line(child: subprocess.Popen[str], expected: str) -> None:
    assert child.stdout is not None
    if not select.select([child.stdout], [], [], _TIMEOUT_SECONDS)[0]:
        child.kill()
        child.wait(timeout=_TIMEOUT_SECONDS)
        assert child.stderr is not None
        pytest.fail(f"child did not report {expected!r}: {child.stderr.read()}")
    assert child.stdout.readline() == f"{expected}\n"


class _FileBackedNixlAgent(FakeNixlAgent):
    def __init__(self) -> None:
        super().__init__(b"guard-md")
        self.name = b"target"
        self.block_remote_writes = False
        self.blocked_remote_writes = 0
        self.backends: dict[str, dict[str, str]] = {}
        self._xfer_backends: dict[int, tuple[str, ...]] = {}

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

    def initialize_xfer(
        self,
        op,
        local_descs,
        remote_descs,
        remote_agent,
        notif_msg=b"",
        backends=None,
    ):
        handle = super().initialize_xfer(
            op,
            local_descs,
            remote_descs,
            remote_agent,
            notif_msg,
            backends,
        )
        self._xfer_backends[handle] = tuple(backends or ())
        return handle

    def transfer(self, handle):
        if not self._xfer_backends[handle]:
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


@pytest.fixture
def live_service(
    tmp_path: Path,
) -> Iterator[tuple[_KVCRService, Callable[[str], subprocess.Popen[str]]]]:
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

    def spawn(program: str) -> subprocess.Popen[str]:
        child = subprocess.Popen(
            [sys.executable, "-c", program],
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


@pytest.mark.parametrize("recovery", ["kept", "given-up"])
def test_request_timeout_during_promotion_then_retry_uses_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery: str,
    live_service: tuple[_KVCRService, Callable[[str], subprocess.Popen[str]]],
) -> None:
    """A retry after promotion is served warm or refused cold, never left hanging."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    first_payload = b"A" * page_size
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
        child = spawn(
            _primary_program(
                service.socket_path,
                g3_path,
                first_payload,
                second_payload,
                primary_port,
            )
        )
        _expect_child_line(child, "claimed")
        assert child.stdin is not None
        child.stdin.write("deposit\n")
        child.stdin.flush()
        _expect_child_line(child, "stable")

        target_control = ZmqPeerControlChannel(
            "127.0.0.1",
            free_port(),
            "127.0.0.1",
        )
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
        assert target_agent.registrations == [
            (
                [(ctypes.addressof(target_memory), len(target_memory), 0, "")],
                "DRAM",
            )
        ]
        now = [0.0]
        target._core._clock = lambda: now[0]
        # The G2 block: a Guard serves what the pool holds and opens no G3.
        key = BlockKey(b"resident-b")
        stalled_destination = (ctypes.c_char * page_size).from_buffer(target_memory)
        target.submit_hint(
            (),
            src=source_endpoint,
            request_id="stalled",
        )
        target.deliver(
            {
                key: _mem_descriptor(
                    ctypes.addressof(stalled_destination), len(stalled_destination)
                )
            },
            request_id="stalled",
        )
        _expect_child_line(child, "in-flight")
        _wait_until(
            lambda: (
                source_endpoint in target._core._remote_fw_dram._metadata_acked_sources
            ),
            timeout=2,
        )

        child.kill()
        child.wait(timeout=_TIMEOUT_SECONDS)
        assert promotion_started.wait(timeout=_TIMEOUT_SECONDS)
        guard = service._registry._pools[0].guard

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
        target.submit_hint((), src=source_endpoint, request_id="retry")
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


def _handback_program(
    socket_path: Path,
    g3_path: Path,
    first_payload: bytes,
    second_payload: bytes,
    control_port: int,
    agent_name: str,
    deposit: bool,
) -> str:
    """A primary that fills the pool, or one that takes it back from a Guard."""
    child = _claiming_child(
        socket_path, g3_path, len(first_payload), control_port, agent_name
    )
    if deposit:
        return (
            child
            + _deposit_source(first_payload, second_payload)
            + textwrap.dedent(
                """
        print("ready", flush=True)
        time.sleep(60)
        """
            )
        )
    return child + textwrap.dedent(
        f"""
        for key, payload in (
            (first_key, {first_payload!r}),
            (second_key, {second_payload!r}),
        ):
            destination = ctypes.create_string_buffer(len(payload))
            operation = kvcr.deliver(
                {{key: _mem_descriptor(
                    ctypes.addressof(destination), len(payload)
                )}}
            )
            result = dict(_poll_until(kvcr, bool))[operation][key]
            assert result.success, key
            assert destination.raw == payload, key
        print("adopted", flush=True)
        time.sleep(60)
        """
    )


def test_replacement_primary_takes_the_cache_back_from_a_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_service: tuple[_KVCRService, Callable[[str], subprocess.Popen[str]]],
) -> None:
    """A Guard serves the dead primary's G2 block until a replacement adopts it."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    first_payload = b"A" * page_size
    second_payload = b"B" * page_size
    g3_path = tmp_path / "g3.data"
    control_port = free_port()
    guard_agent = _FileBackedNixlAgent()
    guard_agent.state = "DONE"
    monkeypatch.setattr(kvcr_progress, "nixl_agent_config", lambda **kwargs: kwargs)
    monkeypatch.setattr(kvcr_progress, "nixl_agent", lambda _name, _config: guard_agent)
    service, spawn = live_service

    def start(agent_name: str, deposit: bool) -> subprocess.Popen[str]:
        return spawn(
            _handback_program(
                service.socket_path,
                g3_path,
                first_payload,
                second_payload,
                control_port,
                agent_name,
                deposit,
            )
        )

    primary = start("primary", deposit=True)
    _expect_child_line(primary, "ready")
    first_guard = service._registry._pools[0].guard

    primary.kill()
    primary.wait(timeout=_TIMEOUT_SECONDS)
    _wait_until(lambda: first_guard._serving, timeout=_TIMEOUT_SECONDS)
    # The G2 half is what a Guard serves; resident-a is spilled to G3.
    assert set(first_guard._core._block_record_map) == {BlockKey(b"resident-b")}

    replacement = start("replacement", deposit=False)
    _expect_child_line(replacement, "adopted")


def _await_marker(
    child: subprocess.Popen[str], marker: str, timeout: float = _TIMEOUT_SECONDS
) -> None:
    """Wait for a line the child meant to send, ignoring what NIXL logs."""
    assert child.stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    # A thread, not select(): TextIOWrapper may already hold the line buffered.
    reader = threading.Thread(target=_drain_lines, args=(child.stdout, lines))
    reader.daemon = True
    reader.start()
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


def _real_nixl_program(
    socket_path: Path,
    g3_path: Path,
    slot_bytes: int,
    g3_bytes: int,
    control_port: int,
    agent_name: str,
    first_payload: bytes,
    second_payload: bytes,
    deposit: bool,
) -> str:
    """A primary that talks to a real NIXL agent over a real G3 file."""
    test_utils = str(Path(__file__).parent)
    body = (
        """
        for key, payload in ((first_key, FIRST), (second_key, SECOND)):
            ctypes.memmove(source, payload, len(payload))
            operation = kvcr.deposit({key: _mem_descriptor(source, len(payload))})
            assert dict(_poll_until(kvcr, bool))[operation][key].success, key
        assert kvcr.query((first_key,)) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
        print("ready", flush=True)
        """
        if deposit
        else """
        for key, payload in ((first_key, FIRST), (second_key, SECOND)):
            ctypes.memset(destination, 0, len(payload))
            operation = kvcr.deliver({key: _mem_descriptor(destination, len(payload))})
            result = dict(_poll_until(kvcr, bool))[operation][key]
            assert result.success, key
            assert ctypes.string_at(destination, len(payload)) == payload, key
        print("adopted", flush=True)
        """
    )
    return textwrap.dedent(
        f"""
        import ctypes
        import sys
        import time
        from pathlib import Path

        sys.path.insert(0, {test_utils!r})
        from _kvcr_test_utils import FakePrimaryPinning, _mem_descriptor, _poll_until
        from kvcr import KVCR, KVCRBindings
        from kvcr.config import (
            FrameworkDramInput,
            G3Options,
            KVCRBackendConfigs,
            KVCRConfig,
            KVCRGuardConfig,
            RemoteFWDramOptions,
        )
        from kvcr.control_channels import ZmqPeerControlChannel
        from kvcr.types import BlockKey, CacheTier, QueryStatus

        FIRST = {first_payload!r}
        SECOND = {second_payload!r}
        first_key = BlockKey(b"resident-a")
        second_key = BlockKey(b"resident-b")

        # Registered with NIXL, and every descriptor this child hands KVCR
        # points inside it: a real agent refuses a transfer that names memory
        # it was never told about.
        framework = ctypes.create_string_buffer({slot_bytes} * 2)
        source = ctypes.addressof(framework)
        destination = ctypes.addressof(framework) + {slot_bytes}

        pinning = FakePrimaryPinning()
        kvcr = KVCR(
            KVCRConfig(
                nixl_agent_name={agent_name!r},
                nixl_listen_port=0,
                inventory_report_interval_ms=0,
            ),
            KVCRBindings(
                pinning.request_pin,
                pinning.poll_pin_results,
                pinning.release_pin,
                framework_control=ZmqPeerControlChannel(
                    "127.0.0.1", {control_port}, "127.0.0.1"
                ),
            ),
            KVCRBackendConfigs(
                framework_dram=FrameworkDramInput(
                    ctypes.addressof(framework), len(framework)
                ),
                g3=G3Options(
                    paths=(Path({str(g3_path)!r}),),
                    capacity_bytes_per_file={g3_bytes},
                    backend="POSIX",
                ),
                remote_fw_dram=RemoteFWDramOptions(eager_ctrl_connect=False),
            ),
            KVCRGuardConfig(
                kvcr_service_socket_path={str(socket_path)!r},
                pool_index=0,
                row_stride={slot_bytes},
                compatibility_digest={_DIGEST!r},
            ),
        )
        {textwrap.indent(textwrap.dedent(body), " " * 8).strip()}
        time.sleep(60)
        """
    )


def test_a_promoted_guard_serves_real_nixl_transfers(
    tmp_path: Path,
    live_service: tuple[_KVCRService, Callable[[str], subprocess.Popen[str]]],
) -> None:
    """With nothing faked, a promoted Guard serves a real UCX read then stands down."""
    # Not a decorator: children import this module, and NIXL logs to their stdout.
    if not _real_nixl_available():
        pytest.skip("no runnable NIXL agent on this machine")
    page_size = os.sysconf("SC_PAGE_SIZE")
    first_payload = b"A" * page_size
    second_payload = b"B" * page_size
    g3_path = tmp_path / "g3.data"
    control_port = free_port()
    service, spawn = live_service

    def start(agent_name: str, deposit: bool) -> subprocess.Popen[str]:
        return spawn(
            _real_nixl_program(
                service.socket_path,
                g3_path,
                page_size,
                page_size * 2,
                control_port,
                agent_name,
                first_payload,
                second_payload,
                deposit,
            )
        )

    primary = start("real-primary", deposit=True)
    _await_marker(primary, "ready", _REAL_NIXL_TIMEOUT_SECONDS)
    guard = service._registry._pools[0].guard

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
    replacement = start("real-replacement", deposit=False)
    _await_marker(replacement, "adopted", _REAL_NIXL_TIMEOUT_SECONDS)
    assert guard._serving is False
