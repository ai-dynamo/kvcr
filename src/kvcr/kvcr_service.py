# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR-Service: the process that owns what outlives a worker.

Today that is shared-memory pools and their recovery Guards; a worker claims one.
"""

import argparse
import contextlib
import functools
import logging
import mmap
import os
import re
import signal
import socket
import socketserver
import stat
import sys
import threading
import time
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from types import FrameType
from typing import Any

from .control_channels import (
    FramedConnection,
    KVCRGuardProtocolError,
    KVCRMsgFramingError,
    KVCRServiceError,
)
from .guard import _Guard, _Lease
from .guard_protocol import (
    _CLAIM_DECODER,
    _PROTOCOL_VERSION,
    _RELEASE_DECODER,
    PidfdLiveness,
    _Claim,
    _Error,
    _Granted,
    _Released,
    _TierConfig,
)
from .memory import (
    _POOL_PREFIX,
    KVCRPoolSpec,
    _KVCRPoolOwner,
    _pool_dir_guard,
    _reclaim_pool_if_orphaned,
    _unlink_if_identity,
)

logger = logging.getLogger(__name__)

_PRIVATE_SOCKET_UMASK = 0o177
_CLIENT_IDLE_TIMEOUT_SECONDS = 30.0
_REGISTRY_TRANSITION_TIMEOUT_SECONDS = 30.0
_STALE_SOCKET_PROBE_SECONDS = 1.0
_DEFAULT_JOURNAL_BYTES = 100 * (1 << 20)
# Matches names produced by memory._pool_filename: "kvcr-<pool_id>-<32 hex>".
_ORPHANED_POOL_NAME = re.compile(rf"{re.escape(_POOL_PREFIX)}-.+-[0-9a-f]{{32}}")
_GB = 1 << 30


class _PoolRegistry:
    """A directory of pools, each owned end to end by its own Guard thread.

    No locks: each pool's mailbox orders its claims, releases and deaths;
    pools share nothing but the refusal flag.
    """

    def __init__(
        self,
        pool_dir: str | os.PathLike[str],
        guard_count: int,
        pool_sizes_bytes: tuple[int, ...],
        journal_bytes: int,
        compatibility_digest: str,
    ) -> None:
        self._pool_dir = Path(pool_dir).resolve()
        if not self._pool_dir.is_dir():
            raise ValueError(f"KVCR pool directory does not exist: {self._pool_dir}")
        self._guard_count = guard_count
        self._pools: dict[int, _Guard] = {}
        self._refusing = threading.Event()
        # Stand-in until the owning server takes over; only it can stop the service.
        self.on_uncontained_failure = functools.partial(
            logger.critical, "Uncontained KVCR pool failure: %s"
        )
        self._purge_orphaned_pools()
        mapping_bytes = journal_bytes + sum(pool_sizes_bytes)
        for rank in range(guard_count):
            try:
                owner = _KVCRPoolOwner.allocate(
                    pool_id=f"pool_{rank}",
                    pool_size_bytes=mapping_bytes,
                    journal_bytes=journal_bytes,
                    pool_dir=self._pool_dir,
                )
                # Built with the pool, not a claim: a Guard that cannot attach its
                # pool is better discovered at startup than when a worker dies.
                try:
                    guard = _Guard(
                        owner.spec,
                        functools.partial(self._guard_failed, rank),
                        compatibility_digest=compatibility_digest,
                        pool_index=rank,
                        owner=owner,
                        refusing=self._refusing.is_set,
                    )
                except BaseException:
                    # Nothing has recorded this pool yet, so the sweep below cannot
                    # reach it and its file would outlive the process.
                    owner.close()
                    raise
                # Recorded before it starts, so a failed preparation is rolled back by
                # the sweep below.
                self._pools[rank] = guard
                guard.start()
            except BaseException:
                self._release_pools()
                raise

    def _release_pools(self) -> None:
        """Give back every pool built so far, for a startup that cannot finish."""
        for pool_index, guard in list(self._pools.items()):
            # Even an interrupt must not stop the sweep: the startup failure is
            # already propagating, and every pool left behind is committed RAM.
            try:
                guard.close()
            except BaseException:
                # The Guard's thread may still hold this pool's mapping, and
                # unlinking under it would fault the process. Leave the file
                # for the next start's purge, and keep the pool visible.
                logger.warning(
                    "Failed to close KVCR Guard for %s; leaving its pool in place",
                    guard._spec.path,
                    exc_info=True,
                )
                continue
            del self._pools[pool_index]

    def _purge_orphaned_pools(self) -> None:
        """Remove pool files no live daemon owns.

        Crash leftovers fill the fixed-size directory and fail the next eager
        allocation. Use is decided by the shared flock, never the filename.
        """
        with _pool_dir_guard(self._pool_dir, exclusive=True):
            for path in sorted(self._pool_dir.glob(f"{_POOL_PREFIX}-*")):
                recognized = _ORPHANED_POOL_NAME.fullmatch(path.name) is not None
                if not recognized or path.is_symlink():
                    continue
                try:
                    size = path.stat().st_size
                    reclaimed = _reclaim_pool_if_orphaned(path)
                except OSError:
                    logger.warning(
                        "Failed to inspect candidate KVCR pool: %s", path, exc_info=True
                    )
                    continue
                if reclaimed:
                    logger.warning(
                        "Reclaimed orphaned KVCR pool from a dead instance: "
                        "%s (%d bytes)",
                        path,
                        size,
                    )
                else:
                    logger.info("Leaving KVCR pool still in use: %s", path)

    def _pool(self, pool_index: int) -> _Guard:
        if not (0 <= pool_index < self._guard_count):
            raise KVCRServiceError(
                f"pool_index {pool_index} is out of range [0, {self._guard_count})"
            )
        guard = self._pools.get(pool_index)
        if guard is None:
            raise KVCRServiceError(f"no claimable KVCR pool {pool_index}")
        return guard

    def claim(
        self,
        pool_index: int,
        tier_config: _TierConfig,
        liveness: PidfdLiveness,
        control_bind: tuple[str, int],
    ) -> "tuple[KVCRPoolSpec, int, _Lease]":
        """Give a pool to a primary, and hand back the endpoint it answers on.

        The refusal check is a fast path only; the grant commits on the pool's
        actor under the same lock refuse_claims reads, so no grant follows it.
        """
        if self._refusing.is_set():
            raise KVCRServiceError("KVCR pool registry is closed")
        return self._pool(pool_index).claim(liveness, tier_config, control_bind)

    def release(self, pool_index: int, lease: "_Lease") -> None:
        self._pool(pool_index).release(lease)

    def abort_grant(self, pool_index: int, lease: "_Lease") -> None:
        """Take back a grant its claimant declared it never served."""
        self._pool(pool_index).abort_grant(lease)

    def refuse_claims(self) -> None:
        """Stop granting pools without waiting for the close path to run.

        Each pool's phase lock is the barrier: after this returns, no grant can
        commit. Pre-barrier grants may still deliver; those leases are fenced.
        """
        self._refusing.set()
        # Snapshot: close() deletes pools from the dict on other threads.
        for guard in list(self._pools.values()):
            with guard._phase_lock:
                pass

    def close(self) -> None:
        """Give every pool back, keeping the first reason one would not go.

        A pool that will not close keeps only its own file and endpoint and
        stays listed, so a later close can try it again; failing that, the
        flock dies with the process and the next start reclaims.
        """
        self._refusing.set()
        failure: BaseException | None = None
        # Tell all pools before waiting on any: a wedged one must not block the rest.
        for guard in self._pools.values():
            try:
                guard.begin_close()
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
        deadline = time.monotonic() + _REGISTRY_TRANSITION_TIMEOUT_SECONDS
        wedged: list[int] = []
        kept: set[int] = set()
        for pool_index in sorted(self._pools):
            try:
                if self._pools[pool_index].finish_close(deadline):
                    wedged.append(pool_index)
                    kept.add(pool_index)
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
                kept.add(pool_index)
        # Wedged pools stay visible; drained ones stay listed until the whole
        # drain finished, so a release racing shutdown is absorbed.
        for pool_index in [index for index in self._pools if index not in kept]:
            del self._pools[pool_index]
        if failure is not None:
            raise failure
        if wedged:
            raise TimeoutError(
                f"timed out waiting for KVCR pool transitions: pools {wedged} leaked"
            )

    def _guard_failed(
        self, pool_index: int, _guard: "_Guard", error: BaseException
    ) -> None:
        """A Guard has stopped being one, which the service cannot survive.

        TODO: no per-pool containment. Its pool can no longer be recovered and may
        still hold an endpoint the service cannot reach. One pool takes the others'
        workers with it; add isolation back if that stops being acceptable.
        """
        logger.critical("KVCR pool %d Guard failed", pool_index)
        self.on_uncontained_failure(error)


class _RequestHandler(socketserver.BaseRequestHandler):
    server: "_ThreadingUnixServer"
    request: socket.socket

    def setup(self) -> None:
        self.channel = FramedConnection(self.request)

    def handle(self) -> None:
        try:
            request = self.channel.receive(_CLAIM_DECODER)
        except (EOFError, OSError):
            return
        except (KVCRGuardProtocolError, KVCRMsgFramingError) as error:
            self._send_error(error)
            return

        liveness: PidfdLiveness | None = None
        grant: "tuple[int, int, _Lease] | None" = None
        try:
            liveness = PidfdLiveness.from_peer_socket(self.request)
            response, grant = self.server.dispatch(request, liveness)
        except KVCRServiceError as error:
            response = _Error(str(error), _PROTOCOL_VERSION)
        except Exception:  # noqa: BLE001 - report internal claim failures
            logger.exception("Unexpected failure while handling KVCR claim")
            response = _Error("internal KVCR service error", _PROTOCOL_VERSION)

        if grant is None:
            if liveness is not None:
                liveness.close()
            with contextlib.suppress(OSError):
                self.channel.send(response)
            return

        pool_index, listener_fd, lease = grant
        try:
            # No timeout: a held connection must wait as long as the lease lives.
            # Inside the try: shutdown may have landed; listener_fd closes either way.
            self.request.settimeout(None)
            self.channel.send_with_fd(response, listener_fd)
        # The lease is already granted and the send may have partially
        # crossed: releasing could double-grant a pool a claimant just mapped.
        # Hold it instead, under the same fencing as any lease: the claimant's
        # death frees it, and a claimant that never saw the grant says so with
        # an unactivated release. EOF alone frees nothing -- a claimant that
        # DID map the pool can drop its connection while still alive.
        except BaseException:
            logger.exception("KVCR grant delivery failed; holding the lease")
        finally:
            with contextlib.suppress(OSError):
                os.close(listener_fd)
        self._await_release(pool_index, lease)

    def _await_release(self, pool_index: int, lease: "_Lease") -> None:
        """Wait for the one message a held connection may send: its release.

        The pool's actor watches the pidfd, not this thread. EOF only ends the
        connection; the lease outlives it, and a death still promotes.
        """
        while True:
            try:
                release = self.channel.receive(_RELEASE_DECODER)
            except (EOFError, OSError):
                return
            except (KVCRGuardProtocolError, KVCRMsgFramingError) as error:
                self._send_error(error)
                continue
            if self._release_or_fail(pool_index, lease, release.activated):
                with contextlib.suppress(OSError):
                    self.channel.send(_Released(_PROTOCOL_VERSION))
            return

    def _release_or_fail(
        self, pool_index: int, lease: "_Lease", activated: bool = True
    ) -> bool:
        try:
            if activated:
                self.server.registry.release(pool_index, lease)
            else:
                # The claimant declared it never served this lease; the Guard
                # it stood down may resume serving.
                self.server.registry.abort_grant(pool_index, lease)
        except KVCRServiceError as error:
            # Registry closing: claimant learns the release did not commit; not fatal.
            self._send_error(error)
            return False
        except BaseException as error:  # noqa: BLE001 - post-grant failure
            self.server.fail(error)
            return False
        return True

    def finish(self) -> None:
        self.server.remove_connection(self.request)

    def _send_error(self, error: Exception) -> None:
        with contextlib.suppress(OSError):
            self.channel.send(_Error(str(error), _PROTOCOL_VERSION))


class _ThreadingUnixServer(
    socketserver.ThreadingMixIn,
    socketserver.UnixStreamServer,
):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        registry: _PoolRegistry,
        compatibility_digest: str,
    ) -> None:
        self.registry = registry
        registry.on_uncontained_failure = self.fail
        self.compatibility_digest = compatibility_digest
        self._fatal_error: BaseException | None = None
        self._fatal_lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        previous_umask = os.umask(_PRIVATE_SOCKET_UMASK)
        try:
            super().__init__(os.fspath(socket_path), _RequestHandler)
        finally:
            os.umask(previous_umask)

    def get_request(self) -> tuple[socket.socket, Any]:
        connection, address = super().get_request()
        connection.settimeout(_CLIENT_IDLE_TIMEOUT_SECONDS)
        self.add_connection(connection)
        return connection, address

    def dispatch(
        self,
        request: _Claim,
        liveness: PidfdLiveness,
    ) -> "tuple[_Granted, tuple[int, int, _Lease]]":
        if request.compatibility_digest != self.compatibility_digest:
            raise KVCRServiceError(
                "KVCR compatibility digest does not match the service"
            )
        spec, listener_fd, lease = self.registry.claim(
            request.pool_index,
            request.tier_config,
            liveness,
            (request.control_host, request.control_port),
        )
        return (
            _Granted(request.pool_index, spec, request.tier_config, _PROTOCOL_VERSION),
            (request.pool_index, listener_fd, lease),
        )

    def fail(self, error: BaseException) -> None:
        """Stop the service, keeping only the first failure.

        Locked: Guard and request threads race here; the first explains the rest.
        """
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = error
        # Before shutdown, not as part of it: a handler already inside claim() would
        # otherwise be granted a pool by a service on its way out.
        self.registry.refuse_claims()
        self.shutdown()

    def add_connection(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._connections.add(connection)

    def remove_connection(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._connections.discard(connection)

    def close_connections(self) -> None:
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                connection.close()


class _KVCRService:
    """Lifecycle wrapper used by the CLI and focused tests."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        pool_dir: str | os.PathLike[str],
        guard_count: int,
        pool_sizes_bytes: tuple[int, ...],
        compatibility_digest: str,
        journal_bytes: int = _DEFAULT_JOURNAL_BYTES,
    ) -> None:
        self.socket_path = Path(socket_path).resolve()
        if not self.socket_path.parent.is_dir():
            raise ValueError(
                f"KVCR socket directory does not exist: {self.socket_path.parent}"
            )
        # One daemon owns a socket path: the deployment runs a single service
        # per pod and clears the path before starting it.
        _unlink_stale_socket(self.socket_path)
        self._registry = _PoolRegistry(
            pool_dir, guard_count, pool_sizes_bytes, journal_bytes, compatibility_digest
        )
        try:
            self._server = _ThreadingUnixServer(
                self.socket_path, self._registry, compatibility_digest
            )
            socket_stat = self.socket_path.stat(follow_symlinks=False)
            self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
        except BaseException:
            self._release_partial_construction()
            raise
        self._closed = False

    def _release_partial_construction(self) -> None:
        """Release whatever __init__ managed to create before it failed."""
        server = getattr(self, "_server", None)
        if server is not None:
            with contextlib.suppress(OSError):
                server.server_close()
        with contextlib.suppress(BufferError, OSError):
            self._registry.close()

    def serve_forever(self) -> None:
        self._server.serve_forever()
        if self._server._fatal_error is not None:
            raise self._server._fatal_error

    def shutdown(self) -> None:
        # Before the accept loop stops, for the same reason.
        self._registry.refuse_claims()
        self._server.shutdown()

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None
        try:
            self._server.close_connections()
        except OSError as error:
            first_error = error
        try:
            self._server.server_close()
        except OSError as error:
            if first_error is None:
                first_error = error
        try:
            self._registry.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
        try:
            _unlink_if_identity(self.socket_path, self._socket_identity)
        except OSError as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error
        self._closed = True


def _unlink_stale_socket(path: Path) -> None:
    """Remove a socket file left behind by a crashed daemon.

    A live daemon accepts the connection, so only a refused connect is stale.
    The unlink is identity-guarded: another daemon may have replaced the path
    between the probe and here, and that socket is live.
    """
    try:
        stat_result = path.lstat()
    except OSError:
        return
    if not stat.S_ISSOCK(stat_result.st_mode):
        return
    identity = (stat_result.st_dev, stat_result.st_ino)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(_STALE_SOCKET_PROBE_SECONDS)
        probe.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        logger.warning("Removing stale KVCR service socket: %s", path)
        _unlink_if_identity(path, identity)
        return
    except OSError as error:
        raise OSError(f"KVCR service socket is not usable: {path}: {error}") from error
    finally:
        probe.close()
    raise OSError(f"another KVCR service is listening on {path}")


class _ShutdownRequested(Exception):
    pass


def _handle_shutdown_signal(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise _ShutdownRequested


def _parse_pool_sizes_gb(value: str) -> tuple[int, ...]:
    try:
        sizes_gb = tuple(Decimal(item) for item in value.split(","))
        if not all(size.is_finite() and size > 0 for size in sizes_gb):
            raise InvalidOperation
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(
            "--pool-sizes-gb must contain positive, finite numbers"
        ) from error
    if any(size > sys.maxsize for size in sizes_gb):
        raise argparse.ArgumentTypeError(
            "--pool-sizes-gb describes an allocation that is too large"
        )
    page = mmap.PAGESIZE
    with localcontext() as context:
        context.prec = max(len(size.as_tuple().digits) for size in sizes_gb) + 10
        sizes_bytes = tuple(int(size * _GB) // page * page for size in sizes_gb)
    if not all(size >= page for size in sizes_bytes):
        raise argparse.ArgumentTypeError(
            "--pool-sizes-gb values must be at least one memory page"
        )
    if _DEFAULT_JOURNAL_BYTES + sum(sizes_bytes) > sys.maxsize:
        raise argparse.ArgumentTypeError(
            "--pool-sizes-gb describes an allocation that is too large"
        )
    return sizes_bytes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the service's command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--guard-count", type=int, required=True)
    parser.add_argument(
        "--pool-sizes-gb",
        type=_parse_pool_sizes_gb,
        dest="pool_sizes_bytes",
        required=True,
    )
    parser.add_argument("--compatibility-digest", required=True)
    args = parser.parse_args(argv)
    if args.guard_count < 1:
        parser.error("--guard-count must be at least 1")
    return args


def main() -> None:
    """Run the standalone KVCR service daemon."""
    args = _parse_args()

    shutdown_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, shutdown_signals)
    old_sigint = signal.signal(signal.SIGINT, _handle_shutdown_signal)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    server: _KVCRService | None = None
    try:
        server = _KVCRService(
            args.socket_path,
            args.pool_dir,
            guard_count=args.guard_count,
            pool_sizes_bytes=args.pool_sizes_bytes,
            compatibility_digest=args.compatibility_digest,
        )
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        logging.basicConfig(level=logging.INFO)
        logger.info(
            "KVCR service ready: socket=%s guards=%d pool_sizes_bytes=%s "
            "journal_bytes=%d",
            args.socket_path,
            args.guard_count,
            args.pool_sizes_bytes,
            _DEFAULT_JOURNAL_BYTES,
        )
        server.serve_forever()
    except _ShutdownRequested:
        pass
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, shutdown_signals)
        try:
            if server is not None:
                server.close()
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


if __name__ == "__main__":
    main()
