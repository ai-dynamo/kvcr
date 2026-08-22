# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR-Service: the process that owns what outlives a worker.

Today that is shared-memory pools; a worker claims one.
"""

import argparse
import contextlib
import logging
import math
import os
import re
import select
import signal
import socket
import socketserver
import stat
import threading
from pathlib import Path
from types import FrameType
from typing import Any

import msgspec

from .control_channels import (
    FramedConnection,
    KVCRGuardProtocolError,
    KVCRMsgFramingError,
    KVCRServiceError,
)
from .guard_protocol import (
    _CLAIM_DECODER,
    _RELEASE_DECODER,
    PidfdLiveness,
    _Claim,
    _Error,
    _Granted,
    _Released,
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
_HOLD_POLL_MILLISECONDS = 1000
_STALE_SOCKET_PROBE_SECONDS = 1.0
# Matches names produced by memory._pool_filename: "kvcr-<pool_id>-<32 hex>".
_ORPHANED_POOL_NAME = re.compile(rf"{re.escape(_POOL_PREFIX)}-.+-[0-9a-f]{{32}}")


class _PoolRegistry:
    def __init__(
        self,
        pool_dir: str | os.PathLike[str],
        pool_count: int,
        pool_size_bytes: int,
    ) -> None:
        self._pool_dir = Path(pool_dir).resolve()
        if not self._pool_dir.is_dir():
            raise ValueError(f"KVCR pool directory does not exist: {self._pool_dir}")
        self._pool_count = pool_count
        self._owners: dict[int, _KVCRPoolOwner] = {}
        self._bindings: dict[int, PidfdLiveness] = {}
        self._row_stride: int | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._purge_orphaned_pools()
        for rank in range(pool_count):
            try:
                self._owners[rank] = _KVCRPoolOwner.allocate(
                    pool_id=f"pool_{rank}",
                    pool_size_bytes=pool_size_bytes,
                    pool_dir=self._pool_dir,
                )
            except BaseException:
                for existing in self._owners.values():
                    try:
                        existing.close()
                    except (BufferError, OSError):
                        logger.warning(
                            "Failed to release KVCR pool %s during allocation rollback",
                            existing.spec.path if existing.spec else "<pending>",
                            exc_info=True,
                        )
                raise

    def _purge_orphaned_pools(self) -> None:
        """Remove pool files no live daemon owns.

        A pool left behind by a crash fills a fixed-size directory and makes
        the next eager allocation fail. Use is decided by the shared flock,
        never the filename, so live daemons and attached workers are untouched.
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

    def claim(
        self,
        pool_index: int,
        row_stride: int,
        liveness: PidfdLiveness,
    ) -> KVCRPoolSpec:
        with self._lock:
            self._ensure_open()
            if not (0 <= pool_index < self._pool_count):
                raise KVCRServiceError(
                    f"pool_index {pool_index} is out of range [0, {self._pool_count})"
                )
            current = self._bindings.get(pool_index)
            if current is not None:
                poller = select.poll()
                poller.register(current.fileno(), select.POLLIN)
                if not any(flags & select.POLLIN for _, flags in poller.poll(0)):
                    raise KVCRServiceError(
                        f"KVCR pool {pool_index} is held by another worker"
                    )
                del self._bindings[pool_index]

            if self._row_stride is not None and row_stride != self._row_stride:
                raise KVCRServiceError(
                    "geometry mismatch for KVCR service: "
                    f"stored row_stride={self._row_stride}; "
                    f"requested row_stride={row_stride}"
                )

            owner = self._owners[pool_index]
            spec = owner.spec
            if spec is None:
                try:
                    spec = owner.finalize(row_stride)
                except msgspec.ValidationError:
                    raise
                except ValueError as error:
                    raise KVCRServiceError(str(error)) from error
                self._row_stride = row_stride
            self._bindings[pool_index] = liveness
            return spec

    def release(self, pool_index: int, liveness: PidfdLiveness) -> None:
        with self._lock:
            self._unbind_locked(pool_index, liveness)

    def holder_died(self, pool_index: int, liveness: PidfdLiveness) -> None:
        with self._lock:
            self._unbind_locked(pool_index, liveness)

    def _unbind_locked(self, pool_index: int, liveness: PidfdLiveness) -> None:
        if self._bindings.get(pool_index) is liveness:
            del self._bindings[pool_index]
        liveness.close()

    def close(self) -> None:
        first_error: Exception | None = None
        with self._lock:
            self._closed = True
            for liveness in self._bindings.values():
                try:
                    liveness.close()
                except OSError as error:
                    if first_error is None:
                        first_error = error
            self._bindings.clear()
            for owner in self._owners.values():
                try:
                    owner.close()
                except (BufferError, OSError) as error:
                    if first_error is None:
                        first_error = error
            self._owners.clear()
        if first_error is not None:
            raise first_error

    def _ensure_open(self) -> None:
        if self._closed:
            raise KVCRServiceError("KVCR pool registry is closed")


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
        pool_index: int | None = None
        try:
            liveness = PidfdLiveness.from_peer_socket(self.request)
            response, pool_index = self.server.dispatch(request, liveness)
        except KVCRServiceError as error:
            response = _Error(str(error))
        except Exception:  # noqa: BLE001 - report internal claim failures
            logger.exception("Unexpected failure while handling KVCR claim")
            response = _Error("internal KVCR service error")

        if pool_index is None:
            if liveness is not None:
                liveness.close()
            with contextlib.suppress(OSError):
                self.channel.send(response)
            return

        assert liveness is not None
        try:
            self.channel.send(response)
        except BaseException:
            self.server.registry.release(pool_index, liveness)
            return
        self._hold(pool_index, liveness)

    def _hold(self, pool_index: int, liveness: PidfdLiveness) -> None:
        try:
            socket_fd = self.request.fileno()
            pidfd = liveness.fileno()
            poller = select.poll()
            poller.register(socket_fd, select.POLLIN | select.POLLHUP | select.POLLERR)
            poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        except (OSError, ValueError) as error:
            self.server.fail(error)
            return

        while True:
            try:
                events = poller.poll(_HOLD_POLL_MILLISECONDS)
            except OSError as error:
                self.server.fail(error)
                return
            if not events:
                try:
                    liveness.fileno()
                except ValueError:
                    return
            for descriptor, flags in events:
                if descriptor == pidfd:
                    if flags & select.POLLIN:
                        self.server.registry.holder_died(pool_index, liveness)
                    else:
                        self.server.fail(
                            OSError(f"pidfd poll returned without POLLIN: {flags:#x}")
                        )
                    return
                if descriptor != socket_fd:
                    continue
                if flags & select.POLLIN:
                    try:
                        self.channel.receive(_RELEASE_DECODER)
                    except (EOFError, OSError):
                        poller.unregister(socket_fd)
                        continue
                    except (KVCRGuardProtocolError, KVCRMsgFramingError) as error:
                        self._send_error(error)
                        continue
                    self.server.registry.release(pool_index, liveness)
                    with contextlib.suppress(OSError):
                        self.channel.send(_Released())
                    return
                if flags & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
                    poller.unregister(socket_fd)

    def finish(self) -> None:
        self.server.remove_connection(self.request)

    def _send_error(self, error: Exception) -> None:
        with contextlib.suppress(OSError):
            self.channel.send(_Error(str(error)))


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
        self.compatibility_digest = compatibility_digest
        self._fatal_error: Exception | None = None
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
    ) -> tuple[_Granted, int]:
        if request.compatibility_digest != self.compatibility_digest:
            raise KVCRServiceError(
                "KVCR compatibility digest does not match the service"
            )
        spec = self.registry.claim(
            request.pool_index,
            request.row_stride,
            liveness,
        )
        return _Granted(request.pool_index, spec), request.pool_index

    def fail(self, error: Exception) -> None:
        self._fatal_error = error
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
        pool_count: int,
        pool_size_bytes: int,
        compatibility_digest: str,
    ) -> None:
        self.socket_path = Path(socket_path).resolve()
        if not self.socket_path.parent.is_dir():
            raise ValueError(
                f"KVCR socket directory does not exist: {self.socket_path.parent}"
            )
        # One daemon owns a socket path: the deployment runs a single service
        # per pod and clears the path before starting it.
        _unlink_stale_socket(self.socket_path)
        self._registry = _PoolRegistry(pool_dir, pool_count, pool_size_bytes)
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
        self._server.shutdown()

    def close(self) -> None:
        if self._closed:
            return
        first_error: Exception | None = None
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
        except (BufferError, OSError) as error:
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the service's command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--pool-count", type=int, required=True)
    parser.add_argument("--pool-size-gb", type=float, required=True)
    parser.add_argument("--compatibility-digest", required=True)
    args = parser.parse_args(argv)

    # Reject nan/inf before converting to int, which would otherwise raise a
    # raw ValueError/OverflowError instead of an argparse usage error.
    if not math.isfinite(args.pool_size_gb) or args.pool_size_gb <= 0:
        parser.error("--pool-size-gb must be a positive, finite number")
    if args.pool_count < 1:
        parser.error("--pool-count must be at least 1")

    args.pool_size_bytes = int(args.pool_size_gb * (1 << 30))
    if args.pool_size_bytes < 1:
        parser.error("--pool-size-gb is too small to reserve a single byte")
    return args


def main() -> None:
    """Run the standalone KVCR service daemon."""
    args = _parse_args()
    pool_size_bytes = args.pool_size_bytes

    shutdown_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, shutdown_signals)
    old_sigint = signal.signal(signal.SIGINT, _handle_shutdown_signal)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    server: _KVCRService | None = None
    try:
        server = _KVCRService(
            args.socket_path,
            args.pool_dir,
            pool_count=args.pool_count,
            pool_size_bytes=pool_size_bytes,
            compatibility_digest=args.compatibility_digest,
        )
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        logging.basicConfig(level=logging.INFO)
        logger.info(
            "KVCR service ready: socket=%s pools=%d pool_size_bytes=%d",
            args.socket_path,
            args.pool_count,
            pool_size_bytes,
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
