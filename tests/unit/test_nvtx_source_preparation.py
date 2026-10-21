# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prepared diagnostics must not move source timing or change callback ownership."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from _nvtx_test_utils import payloads, spans

from kvcr import _nvtx
from kvcr.remote_fw_dram import _SourceWriteOp, _SourceWriteState


def prepared_source(tracer, *, reason=_nvtx.Reason.UNKNOWN):
    source = SimpleNamespace(ordered_keys=(b"key",), target_incarnation="boot")
    backend = SimpleNamespace(
        _kvcr=SimpleNamespace(
            _clock=lambda: 0,
            _timer=lambda: 0,
            _descriptor_bytes=Mock(return_value=16),
            _memory_regions=({}, {}),
        ),
        _dangling_ops=SimpleNamespace(
            source_writes={("target", 7): SimpleNamespace(cancel_requested=False)},
            check_source_progress=lambda: True,
            poll_source=lambda *args, **kwargs: None,
            finish_source=Mock(),
        ),
        _send_write_done=Mock(),
        _record_progress_duration=Mock(),
        _route_generation={"target": 0},
        _options=SimpleNamespace(backend=None),
        _telemetry_enabled=False,
    )
    refs = ((SimpleNamespace(framework=True, label="region"),),)
    op = _SourceWriteOp(
        op_id=("source", 3),
        keys=frozenset(source.ordered_keys),
        started_at=0,
        deadline=100,
        state=_SourceWriteState.READY_TO_WRITE,
        remote_agent="target",
        op_handle=7,
        source_keys=source.ordered_keys,
        src_descriptors=refs,
        dst_descriptors=refs,
        route=("target", 0),
        trace=tracer.prepare_source(source, reason),
        _backend=backend,
    )
    progress = SimpleNamespace(
        submit_transfer=Mock(return_value=(1, True)),
        _active_transfers={1: SimpleNamespace(outcome=None)},
    )
    return op, progress


@pytest.mark.parametrize("level", ["low", "medium"])
def test_source_context_and_sampling_wait_for_progress(recording, monkeypatch, level):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", level)
    op, progress = prepared_source(_nvtx.create_tracer())
    sample = op._backend._kvcr._descriptor_bytes
    assert not recording.events
    sample.assert_not_called()
    results = []
    worker = threading.Thread(
        target=lambda: results.append(op.progress(progress, None))
    )
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results == [(False, True)]
    assert sample.call_count == (level == "medium")
    context = payloads(recording, "request.context")[0]
    assert int(context["requested_blocks"]) == 1
    assert int(context["selected_bytes"]) == (16 if level == "medium" else -1)
    assert (
        int(context["target_incarnation_hi"]),
        int(context["target_incarnation_lo"]),
    ) == _nvtx.identity("boot")
    assert spans(recording, "source.write")[0][3] == worker.ident
    assert spans(recording, "request.context", "mark")[0][3] == worker.ident
    assert 0 in recording.ranges
    assert op.progress(progress, None) == (False, False)
    assert len(spans(recording, "source.write")) == 1


def test_close_before_first_progress_has_no_annotations(recording):
    op, progress = prepared_source(_nvtx.create_tracer(), reason=_nvtx.Reason.DEADLINE)
    assert op.close(progress)
    op._backend._dangling_ops.finish_source.assert_called_once_with(op)
    op._backend._send_write_done.assert_called_once_with(progress, "target", 7, False)
    assert not recording.events
    assert not recording.ranges


@pytest.mark.parametrize("failure", ["diagnostics", "start"])
def test_source_annotation_failure_is_not_retried_on_poll(
    recording, monkeypatch, failure
):
    monkeypatch.setenv("KVCR_NVTX_LEVEL", "medium")
    tracer = _nvtx.create_tracer()
    op, progress = prepared_source(tracer)
    activate = Mock(wraps=tracer.source)
    monkeypatch.setattr(tracer, "source", activate)
    if failure == "diagnostics":
        op._backend._kvcr._descriptor_bytes.side_effect = ValueError("diagnostics")
    else:
        recording.fail = "start"
    assert op.progress(progress, None) == (False, True)
    recording.fail = None
    for _ in range(3):
        assert op.progress(progress, None) == (False, False)
    assert activate.call_count == 1
    progress.submit_transfer.assert_called_once()
    assert not spans(recording, "source.write")


@pytest.mark.parametrize("failure", [None, "push", "pop"])
@pytest.mark.parametrize("raises", [False, True])
def test_framework_callback_runs_once_and_preserves_exception(
    recording, failure, raises
):
    trace = _nvtx.create_tracer().begin(1)
    recording.fail = failure
    error = ValueError("framework")
    callback = Mock(side_effect=error) if raises else Mock(return_value=0)
    if raises:
        with pytest.raises(ValueError) as caught:
            trace.call_framework(callback, (b"key",))
        assert caught.value is error
    else:
        assert trace.call_framework(callback, (b"key",)) == 0
    callback.assert_called_once_with((b"key",))
    assert not recording.stack
