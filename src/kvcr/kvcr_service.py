# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KVCR-Service: the process that owns what outlives a worker.

Today that is shared-memory pools and their recovery Guards; a worker claims one.
"""

import argparse
import contextlib
import functools
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
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

from .control_channels import (
    FramedConnection,
    KVCRGuardProtocolError,
    KVCRMsgFramingError,
    KVCRServiceError,
    ZmqPeerControlChannel,
)
from .guard import _Guard
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
from .recovery_journal import RecoveryMirrorError

logger = logging.getLogger(__name__)

_PRIVATE_SOCKET_UMASK = 0o177
_CLIENT_IDLE_TIMEOUT_SECONDS = 30.0
_HOLD_POLL_MILLISECONDS = 1000
_REGISTRY_TRANSITION_TIMEOUT_SECONDS = 30.0
_STALE_SOCKET_PROBE_SECONDS = 1.0
_DEFAULT_JOURNAL_BYTES = 100 * (1 << 20)
# Matches names produced by memory._pool_filename: "kvcr-<pool_id>-<32 hex>".
_ORPHANED_POOL_NAME = re.compile(rf"{re.escape(_POOL_PREFIX)}-.+-[0-9a-f]{{32}}")


def _log_uncontained_failure(error: BaseException) -> None:
    """Stand-in until a server claims this registry."""
    logger.critical("Uncontained KVCR pool failure: %s", error)


@dataclass
class _PoolDescriptor:
    """Everything the service knows about one pool.

    One record rather than a map per attribute, so a holder, a listener and a
    Guard cannot disagree about which pool they belong to.
    """

    owner: _KVCRPoolOwner
    guard: "_Guard"
    holder: PidfdLiveness | None = None
    listener: socket.socket | None = None
    # The address as the claimant asked for it: getsockname() answers
    # numerically and would reject every alias of the same address.
    bind: tuple[str, int] | None = None
    in_transition: bool = False

    def close_holder(self) -> None:
        if self.holder is not None:
            self.holder.close()
            self.holder = None

    def close_listener(self) -> None:
        if self.listener is not None:
            self.listener.close()
            self.listener = None
            self.bind = None


class _PoolRegistry:
    def __init__(
        self,
        pool_dir: str | os.PathLike[str],
        pool_count: int,
        pool_size_bytes: int,
        journal_bytes: int,
        compatibility_digest: str,
    ) -> None:
        self._pool_dir = Path(pool_dir).resolve()
        if not self._pool_dir.is_dir():
            raise ValueError(f"KVCR pool directory does not exist: {self._pool_dir}")
        self._pool_count = pool_count
        self._compatibility_digest = compatibility_digest
        self._pools: dict[int, _PoolDescriptor] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        # Replaced by the owning server: only it can stop the service.
        self.on_uncontained_failure: Callable[[BaseException], None] = (
            _log_uncontained_failure
        )
        self._purge_orphaned_pools()
        for rank in range(pool_count):
            try:
                owner = _KVCRPoolOwner.allocate(
                    pool_id=f"pool_{rank}",
                    pool_size_bytes=pool_size_bytes,
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
                    )
                except BaseException:
                    # Nothing has recorded this pool yet, so the sweep below cannot
                    # reach it and its file would outlive the process.
                    owner.close()
                    raise
                # Recorded before it starts, so a failed preparation is rolled back by
                # the sweep below.
                self._pools[rank] = _PoolDescriptor(owner, guard)
                guard.start()
            except BaseException:
                self._release_pools_locked()
                raise

    def _release_pools_locked(self) -> None:
        """Give back every pool built so far, for a startup that cannot finish."""
        for entry in self._pools.values():
            # Even an interrupt must not stop the sweep: the startup failure is
            # already propagating, and every pool left behind is committed RAM.
            try:
                entry.guard.close()
            except BaseException:
                # The Guard's thread may still hold this pool's mapping, and
                # unlinking under it would fault the process. Leave the file;
                # the next start's purge reclaims it once the process is gone.
                logger.warning(
                    "Failed to close KVCR Guard for %s during rollback; "
                    "leaving its pool in place",
                    entry.owner.spec.path,
                    exc_info=True,
                )
                continue
            try:
                entry.owner.close()
            except (BufferError, OSError):
                logger.warning(
                    "Failed to release KVCR pool %s during allocation rollback",
                    entry.owner.spec.path,
                    exc_info=True,
                )
        self._pools.clear()

    def _purge_orphaned_pools(self) -> None:
        """Remove pool files no live daemon owns.

        A pool left by a crash fills a fixed-size directory and fails the next eager
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

    def claim(
        self,
        pool_index: int,
        tier_config: _TierConfig,
        liveness: PidfdLiveness,
        control_bind: tuple[str, int],
    ) -> KVCRPoolSpec:
        """Give a pool to a primary, and tell the pool's Guard about it."""
        with self._condition:
            self._ensure_open()
            if not (0 <= pool_index < self._pool_count):
                raise KVCRServiceError(
                    f"pool_index {pool_index} is out of range [0, {self._pool_count})"
                )
            pool = self._pools[pool_index]
            if pool.in_transition:
                raise KVCRServiceError(f"KVCR pool {pool_index} is busy")
            current = pool.holder
            if current is not None:
                poller = select.poll()
                poller.register(current.fileno(), select.POLLIN)
                if not any(flags & select.POLLIN for _, flags in poller.poll(0)):
                    raise KVCRServiceError(
                        f"KVCR pool {pool_index} is held by another worker"
                    )
                # Dead but not yet free: its watcher is the sole promotion authority.
                raise KVCRServiceError(f"KVCR pool {pool_index} is busy")
            # Layout, not use: the claimant and the Guard each derive geometry from the
            # same numbers.
            spec = pool.owner.spec
            bound_here = pool.listener is None
            listener = self._bind_control_listener_locked(
                pool, pool_index, control_bind
            )
            pool.in_transition = True

        # Unlocked: adopting blocks, and every other pool's claim, release and
        # death handling needs this lock in the meantime.
        try:
            # from_shared_listener detaches what it is given, so the Guard gets a
            # duplicate and the service keeps the original. Closed here if never taken:
            # a stray dup of a listening socket keeps the address bound.
            duplicate = socket.socket(fileno=os.dup(listener.fileno()))
            try:
                control = ZmqPeerControlChannel.from_shared_listener(duplicate)
            except BaseException:
                duplicate.close()
                raise
            pool.guard.adopt(control, tier_config)
        except BaseException as error:
            unbind_error: BaseException | None = None
            with self._condition:
                try:
                    if bound_here and pool.holder is None:
                        # This claim chose the address and then failed, so keeping it
                        # would refuse the retry as an endpoint move.
                        try:
                            pool.listener.close()
                        except BaseException as close_error:  # noqa: BLE001
                            unbind_error = close_error
                        pool.listener = None
                        pool.bind = None
                    if unbind_error is not None:
                        # Latched here rather than by the escalation below: the pool
                        # is free and unbound the moment this lock drops, and nothing
                        # may claim it on an address that would not close.
                        self._closed = True
                finally:
                    # Including an interrupt landing mid-cleanup: a pool left in
                    # transition is one no later claim can have, and one shutdown
                    # waits out.
                    self._finish_transition_locked(pool_index)
            if unbind_error is not None:
                # Outside the lock, which refuse_claims needs. An address that will
                # not close is one the service can neither reach nor hand out.
                self._guard_failed(pool_index, pool.guard, unbind_error)
            if isinstance(error, (ValueError, RecoveryMirrorError)):
                raise KVCRServiceError(str(error)) from error
            raise

        with self._condition:
            # Recorded before the check: the pool has already changed hands, so a close
            # racing this must see the holder rather than find it in transition.
            pool.holder = liveness
            self._finish_transition_locked(pool_index)
            self._ensure_open()
        return spec

    def _begin_holder_transition(
        self, pool_index: int, liveness: PidfdLiveness
    ) -> bool:
        """Reserve the transition this lease is ending through, if it owns one.

        A release and a holder's death start the same way: the caller may be reporting
        a lease the registry has already replaced, and a stale one owns nothing.
        """
        with self._condition:
            pool = self._pools.get(pool_index)
            if pool is None or pool.holder is not liveness:
                stale = True
            elif self._closed or pool.in_transition:
                return False
            else:
                stale = False
                pool.in_transition = True
        if not stale:
            return True
        liveness.close()
        return False

    def release(self, pool_index: int, liveness: PidfdLiveness) -> None:
        """End a lease. The pool keeps its Guard, and the Guard its records."""
        if not self._begin_holder_transition(pool_index, liveness):
            return
        pool = self._pools[pool_index]
        try:
            pool.guard.release()
        except BaseException as error:
            # Escalated before the pool is exposed: a Guard that failed here may
            # have left a partial handback, and the next claimant could take it.
            self._guard_failed(pool_index, pool.guard, error)
            raise
        finally:
            liveness.close()
            with self._condition:
                if pool.holder is liveness:
                    pool.holder = None
                self._finish_transition_locked(pool_index)

    def abort_grant(self, pool_index: int, liveness: PidfdLiveness) -> None:
        """Take back a grant whose delivery failed; the claimant never saw it."""
        if not self._begin_holder_transition(pool_index, liveness):
            return
        pool = self._pools[pool_index]
        try:
            pool.guard.abort_grant()
        except BaseException as error:
            self._guard_failed(pool_index, pool.guard, error)
            raise
        finally:
            liveness.close()
            with self._condition:
                if pool.holder is liveness:
                    pool.holder = None
                self._finish_transition_locked(pool_index)

    def holder_died(self, pool_index: int, liveness: PidfdLiveness) -> None:
        """Promote this pool's Guard in place of the primary that died.

        A promotion that fails takes the service with it: the pool has inherited the
        dead primary's endpoint and half-adopted its records.
        """
        if not self._begin_holder_transition(pool_index, liveness):
            return
        pool = self._pools[pool_index]
        try:
            pool.guard.promote_after_death()
        except BaseException as error:
            # Escalated before the pool is exposed: a Guard that failed here may
            # have left a partial handback, and the next claimant could take it.
            self._guard_failed(pool_index, pool.guard, error)
            raise
        finally:
            liveness.close()
            with self._condition:
                if pool.holder is liveness:
                    pool.holder = None
                self._finish_transition_locked(pool_index)

    def refuse_claims(self) -> None:
        """Stop granting pools without waiting for the close path to run."""
        with self._condition:
            self._closed = True

    def is_closed(self) -> bool:
        """Whether this registry has stopped granting pools."""
        with self._condition:
            return self._closed

    def close(self) -> None:
        """Release every pool, keeping the first reason one would not go.

        Pools are independent, so one that will not close keeps only its own file and
        endpoint. Nothing retries this: the flock dies with the process, and the next
        service to start reclaims what is left.
        """
        deadline = time.monotonic() + _REGISTRY_TRANSITION_TIMEOUT_SECONDS
        with self._condition:
            self._closed = True
            while any(entry.in_transition for entry in self._pools.values()):
                if deadline - time.monotonic() <= 0:
                    break
                self._condition.wait(deadline - time.monotonic())
            wedged = sorted(
                index for index, entry in self._pools.items() if entry.in_transition
            )

        # No lock below: claims are refused and both endings return untouched.
        failure: BaseException | None = None
        for pool_index in sorted(self._pools):
            if pool_index in wedged:
                continue
            pool = self._pools[pool_index]
            try:
                # Only this gates the rest: a Guard that will not close may still be
                # serving out of the mapping.
                pool.guard.close()
            except BaseException as error:  # noqa: BLE001 - raised below
                failure = failure or error
                continue
            for step in (pool.close_holder, pool.close_listener, pool.owner.close):
                try:
                    step()
                except BaseException as error:  # noqa: BLE001 - raised below
                    failure = failure or error
            del self._pools[pool_index]
        if failure is not None:
            raise failure
        if wedged:
            raise TimeoutError(
                "timed out waiting for KVCR pool transitions: "
                f"pools {wedged} were left unclosed and leaked their resources"
            )

    def _bind_control_listener_locked(
        self, pool: "_PoolDescriptor", pool_index: int, control_bind: tuple[str, int]
    ) -> socket.socket:
        """The address this pool answers on, bound once and never moved.

        A Guard inherits the endpoint its primary used, so a claim naming a different
        one is refused rather than migrated.
        """
        if pool.listener is not None:
            if pool.bind != control_bind:
                assert pool.bind is not None
                raise KVCRServiceError(
                    f"KVCR pool {pool_index} answers on "
                    f"{pool.bind[0]}:{pool.bind[1]} and cannot be moved to "
                    f"{control_bind[0]}:{control_bind[1]}"
                )
            return pool.listener
        try:
            listener = socket.create_server(control_bind)
        except OSError as error:
            raise KVCRServiceError(
                f"KVCR pool {pool_index} control listener "
                f"{control_bind[0]}:{control_bind[1]} is unavailable: {error}"
            ) from error
        pool.listener = listener
        pool.bind = control_bind
        return listener

    def control_listener_fd(self, pool_index: int) -> int | None:
        with self._condition:
            pool = self._pools.get(pool_index)
            if pool is None or pool.listener is None:
                return None
            listener = pool.listener
            return os.dup(listener.fileno())

    def _guard_failed(
        self, pool_index: int, guard: "_Guard", error: BaseException
    ) -> None:
        """A Guard has stopped being one, which the service cannot survive.

        TODO: no per-pool containment. Its pool can no longer be recovered and may
        still hold an endpoint the service cannot reach. One pool takes the others'
        workers with it; add isolation back if that stops being acceptable.
        """
        del guard
        logger.critical("KVCR pool %d Guard failed", pool_index)
        self.on_uncontained_failure(error)

    def _finish_transition_locked(self, pool_index: int) -> None:
        self._pools[pool_index].in_transition = False
        self._condition.notify_all()

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
            response = _Error(str(error), _PROTOCOL_VERSION)
        except Exception:  # noqa: BLE001 - report internal claim failures
            logger.exception("Unexpected failure while handling KVCR claim")
            response = _Error("internal KVCR service error", _PROTOCOL_VERSION)

        if pool_index is None:
            if liveness is not None:
                liveness.close()
            with contextlib.suppress(OSError):
                self.channel.send(response)
            return

        assert liveness is not None
        try:
            self._deliver_grant(pool_index, response)
        # The lease is already granted: whatever escapes delivery, revoke it
        # or the pool stays held by a claimant that never heard it won.
        except BaseException:
            logger.exception("KVCR grant delivery failed; revoking the lease")
            self._abort_or_fail(pool_index, liveness)
            return
        self._hold(pool_index, liveness)

    def _deliver_grant(self, pool_index: int, response: "_Granted | _Error") -> None:
        """Send a grant, with the endpoint it promises when it promises one."""
        listener_fd: int | None = None
        try:
            if isinstance(response, _Granted):
                listener_fd = self.server.registry.control_listener_fd(pool_index)
                if listener_fd is None:
                    # The grant says a Guard stands behind this endpoint; without the
                    # descriptor the claimant would serve its own.
                    raise KVCRServiceError(
                        f"KVCR pool {pool_index} was granted with a Guard but "
                        "has no control listener to hand over"
                    )
            if listener_fd is None:
                self.channel.send(response)
            else:
                self.channel.send_with_fd(response, listener_fd)
        finally:
            if listener_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(listener_fd)

    def _hold(self, pool_index: int, liveness: PidfdLiveness) -> None:
        try:
            socket_fd = self.request.fileno()
            pidfd = liveness.fileno()
            poller = select.poll()
            poller.register(socket_fd, select.POLLIN | select.POLLHUP | select.POLLERR)
            poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        except (OSError, ValueError) as error:
            self._fail_unless_closed(error)
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
            ready = dict(events)
            socket_flags = ready.get(socket_fd, 0)
            if socket_flags & select.POLLIN:
                try:
                    self.channel.receive(_RELEASE_DECODER)
                except (EOFError, OSError):
                    poller.unregister(socket_fd)
                    socket_flags = 0
                except (KVCRGuardProtocolError, KVCRMsgFramingError) as error:
                    self._send_error(error)
                else:
                    if not self._release_or_fail(pool_index, liveness):
                        return
                    with contextlib.suppress(OSError):
                        self.channel.send(_Released(_PROTOCOL_VERSION))
                    return
            pidfd_flags = ready.get(pidfd, 0)
            if pidfd_flags & select.POLLIN:
                try:
                    self.server.registry.holder_died(pool_index, liveness)
                except BaseException as error:
                    self.server.fail(error)
                return
            if pidfd_flags & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
                self._fail_unless_closed(
                    OSError(f"pidfd poll returned without POLLIN: {pidfd_flags:#x}")
                )
                return
            if socket_flags & (select.POLLHUP | select.POLLERR | select.POLLNVAL):
                poller.unregister(socket_fd)

    def _fail_unless_closed(self, error: BaseException) -> None:
        """Report a broken holder descriptor, unless shutdown is what broke it.

        Closing the registry closes each holder's pidfd without waiting for the
        thread watching it, which that thread sees as a descriptor that died. Both
        the setup and the poll notice it, and neither has anything to report:
        failing here would latch a cause that hides whatever stopped the service.
        """
        if self.server.registry.is_closed():
            return
        self.server.fail(error)

    def _abort_or_fail(self, pool_index: int, liveness: PidfdLiveness) -> None:
        try:
            self.server.registry.abort_grant(pool_index, liveness)
        except BaseException as error:  # noqa: BLE001 - post-grant failure
            self.server.fail(error)

    def _release_or_fail(
        self,
        pool_index: int,
        liveness: PidfdLiveness,
    ) -> bool:
        try:
            self.server.registry.release(pool_index, liveness)
        except BaseException as error:
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
    ) -> tuple[_Granted, int]:
        if request.compatibility_digest != self.compatibility_digest:
            raise KVCRServiceError(
                "KVCR compatibility digest does not match the service"
            )
        spec = self.registry.claim(
            request.pool_index,
            request.tier_config,
            liveness,
            (request.control_host, request.control_port),
        )
        return (
            _Granted(
                request.pool_index,
                spec,
                request.tier_config,
                _PROTOCOL_VERSION,
            ),
            request.pool_index,
        )

    def fail(self, error: BaseException) -> None:
        """Stop the service, keeping the failure that started it.

        A fatal failure cascades and the endings it triggers report failures of their
        own. The first explains the rest, so it is kept -- under a lock, because Guard
        and request threads both arrive here.
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
        pool_count: int,
        pool_size_bytes: int,
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
            pool_dir,
            pool_count,
            pool_size_bytes,
            journal_bytes,
            compatibility_digest,
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
    if args.pool_size_bytes <= _DEFAULT_JOURNAL_BYTES:
        # The journal is carved out of this size, not added to it, so a pool this small
        # has no room to cache anything.
        parser.error(
            "--pool-size-gb must exceed the "
            f"{_DEFAULT_JOURNAL_BYTES >> 20} MiB journal each pool reserves"
        )
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
            "KVCR service ready: socket=%s pools=%d pool_size_bytes=%d "
            "journal_bytes=%d",
            args.socket_path,
            args.pool_count,
            pool_size_bytes,
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
