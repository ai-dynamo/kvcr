# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hint dispatch tests."""

import logging
from typing import ClassVar

import msgspec
import pytest
from _kvcr_test_utils import (
    FakeBytesControl,
    FakeNixlAgent,
    FakePrimaryPinning,
    _HintAction,
    _HintEnvelope,
    _new_kvcr,
    _record_hint_results,
)

from kvcr import STATE_METRIC
from kvcr.config import KVCRConfig
from kvcr.hints.dispatcher import _HINT_TYPES, _LOGGED_HANDLER_ERRORS, HintDispatcher
from kvcr.hints.parser import _LOGGED_HINT_ISSUES, HintType
from kvcr.hints.results import HintResult, HintStatus


@pytest.fixture(autouse=True)
def _reset_once_logs():
    # Once-per-issue warnings are process-global; reset so tests that count
    # warnings do not depend on run order.
    _LOGGED_HANDLER_ERRORS.clear()
    _LOGGED_HINT_ISSUES.clear()


# Dispatch infrastructure tests use stub action types; no production handler.


class _StubHint(msgspec.Struct, frozen=True):
    ACTION_TYPE: ClassVar[str] = "kv.test"

    action_id: str | None


class _StubHandler:
    """Inline handler that accepts every hint except those listed in ``fail``."""

    action_type = "kv.test"

    def __init__(self, core, dispatcher) -> None:
        self.core = core
        self.dispatcher = dispatcher
        self.submitted: list[_StubHint] = []
        self.close_calls = 0
        self.fail: set[str] = set()

    def submit(self, hint, request_id, message_id) -> None:
        if hint.action_id in self.fail:
            raise RuntimeError("submit boom")
        self.submitted.append(hint)
        self.dispatcher.report(
            HintResult(
                action_id=hint.action_id,
                action_type=self.action_type,
                status=HintStatus.ACCEPTED,
                request_id=request_id,
                message_id=message_id,
            )
        )

    def close(self) -> None:
        self.close_calls += 1


_STUB_TYPE = HintType("kv.test", "1.0", lambda i, p: _StubHint(i), _StubHandler)


def _envelope(*hints: tuple[str, str], message_id: str = "m1") -> _HintEnvelope:
    return _HintEnvelope(
        "0.1", message_id, [_HintAction(i, t, "1.0", {}) for i, t in hints]
    )


def _tests(*ids: str) -> _HintEnvelope:
    return _envelope(*((i, "kv.test") for i in ids))


def _stub_kvcr(monkeypatch, **config_fields):
    config = KVCRConfig(
        nixl_agent_name="target", pool_layouts=[("", 16)], **config_fields
    )
    # Handlers come from the hint type table, instantiated per core.
    monkeypatch.setattr("kvcr.hints.dispatcher._HINT_TYPES", (*_HINT_TYPES, _STUB_TYPE))
    kvcr = _new_kvcr(
        FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl(), config=config
    )
    handler = kvcr._core._hints_dispatcher._handlers["kv.test"]
    return kvcr, handler, _record_hint_results(monkeypatch, kvcr)


def _take(results) -> list[tuple[str | None, HintStatus, str | None]]:
    """Statuses reported since the previous take."""
    taken = [(r.action_id, r.status, r.reason) for r in results]
    results.clear()
    return taken


def test_dispatcher_owns_the_hint_type_table(monkeypatch):
    # Core names no handler: the built-in handler comes from the dispatcher's
    # table, and a type added to the table appears in a fresh dispatcher
    # without any core change.
    kvcr = _new_kvcr(FakeNixlAgent(), FakePrimaryPinning(), FakeBytesControl())
    assert set(kvcr._core._hints_dispatcher._handlers) == {"kv.fetch"}

    built_for: list[tuple[object, object]] = []

    def factory(core, dispatcher):
        built_for.append((core, dispatcher))
        return _StubHandler(core, dispatcher)

    monkeypatch.setattr(
        "kvcr.hints.dispatcher._HINT_TYPES",
        (*_HINT_TYPES, HintType("kv.test", "1.0", lambda i, p: _StubHint(i), factory)),
    )
    dispatcher = HintDispatcher(kvcr._core)
    assert set(dispatcher._handlers) == {"kv.fetch", "kv.test"}
    assert built_for == [(kvcr._core, dispatcher)]


def test_submit_routes_to_handler_and_rejects_unregistered_types(monkeypatch):
    kvcr, handler, results = _stub_kvcr(monkeypatch)
    kvcr.submit_hint(
        _envelope(("a", "kv.test"), ("b", "kv.unhandled")), request_id="req"
    )
    assert handler.submitted == [_StubHint("a")]
    # The parser's rejections are reported before the dispatched hints.
    assert [(r.action_id, r.action_type, r.status, r.reason) for r in results] == [
        ("b", "kv.unhandled", HintStatus.REJECTED, "unsupported_action_type"),
        ("a", "kv.test", HintStatus.ACCEPTED, None),
    ]
    assert all((r.request_id, r.message_id) == ("req", "m1") for r in results)


def test_handlers_close_only_once_the_loop_is_quiescent(monkeypatch):
    kvcr, handler, _ = _stub_kvcr(monkeypatch)
    monkeypatch.setattr(kvcr._core, "is_quiescent", lambda: False)
    with pytest.raises(RuntimeError, match="not quiescent"):
        kvcr.close()
    # Native operations may still reference backend state, so handlers (which
    # may own such state) stay open until quiescence is proven.
    assert handler.close_calls == 0
    monkeypatch.undo()
    kvcr.close()
    assert handler.close_calls == 1
    kvcr.close()
    assert handler.close_calls == 1


def test_handler_close_failure_does_not_skip_backend_cleanup(monkeypatch):
    kvcr, handler, _ = _stub_kvcr(monkeypatch)
    cleaned: list[str] = []
    monkeypatch.setattr(
        kvcr._core._remote_fw_dram, "close_main", lambda: cleaned.append("remote")
    )

    def boom() -> None:
        raise RuntimeError("close boom")

    handler.close = boom
    kvcr.close()  # the dispatcher isolates handler close errors
    assert cleaned == ["remote"]


def test_handler_exception_rejects_only_its_hint(monkeypatch, caplog):
    kvcr, handler, results = _stub_kvcr(monkeypatch)
    handler.fail = {"a", "c"}
    with caplog.at_level(logging.WARNING, logger="kvcr.hints.dispatcher"):
        kvcr.submit_hint(_tests("a", "b", "c"), request_id="req")
    # The healthy sibling right after a failing hint is still dispatched.
    assert handler.submitted == [_StubHint("b")]
    assert all((r.request_id, r.message_id) == ("req", "m1") for r in results)
    assert _take(results) == [
        ("a", HintStatus.REJECTED, "handler_error"),
        ("b", HintStatus.ACCEPTED, None),
        ("c", HintStatus.REJECTED, "handler_error"),
    ]
    warnings = [r for r in caplog.records if r.name == "kvcr.hints.dispatcher"]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


def test_progress_failure_propagates_from_submit(monkeypatch):
    kvcr, handler, results = _stub_kvcr(monkeypatch)
    kvcr._core._progress._failure = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        kvcr.submit_hint(_tests("a"), request_id="req")
    with pytest.raises(RuntimeError, match="boom"):
        kvcr._core._hints_dispatcher.submit(_tests("b"), "req")
    assert handler.submitted == []
    assert results == []
    kvcr._core._progress._failure = None


def test_progress_failure_during_dispatch_is_not_a_handler_error(monkeypatch):
    kvcr, handler, results = _stub_kvcr(monkeypatch)

    def fail_then_raise(hint, request_id, message_id) -> None:
        # What the remote store does: the progress thread died, so its submit
        # re-raises the stored failure through the handler.
        kvcr._core._progress._failure = RuntimeError("boom")
        kvcr._core._progress.raise_if_failed()

    handler.submit = fail_then_raise
    with pytest.raises(RuntimeError, match="boom"):
        kvcr.submit_hint(_tests("a"), request_id="req")
    assert results == []
    kvcr._core._progress._failure = None


def test_results_emit_no_metric(monkeypatch):
    # Integrations forward KVCR stats and may reject names they do not
    # declare (vLLM asserts), so hint results must not add a metric.
    kvcr, _, results = _stub_kvcr(monkeypatch, enable_telemetry=True)
    kvcr.get_stats()  # start from an empty snapshot
    kvcr.submit_hint(_envelope(("x", "kv.whatever-1234"), ("a", "kv.test")))
    assert [(r.action_type, r.status) for r in results] == [
        ("kv.whatever-1234", HintStatus.REJECTED),
        ("kv.test", HintStatus.ACCEPTED),
    ]
    assert {name for _, name, _, _ in kvcr.get_stats().records} == {STATE_METRIC}
