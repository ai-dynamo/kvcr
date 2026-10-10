# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""OpenAI-compatible completions client that records every request and reply."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .common import TokenSample, save_json, utc_now

EXTRA_FIELDS = [
    "worker_id",
    "prompt_token_ids",
    "completion_token_ids",
    "detailed_finish_reason",
]


@dataclass
class Response:
    stage: str
    started_at: str
    finished_at: str
    latency_seconds: float
    http_status: int | None
    body: dict[str, Any] | None
    error: str | None
    request_file: str
    response_file: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            self.http_status == 200 and isinstance(self.body, dict) and not self.error
        )


@dataclass
class StreamResponse:
    """One streamed completion with client-side timing and streamed token IDs.

    Times are monotonic seconds relative to the moment the request was sent.
    Token IDs come from nvext.completion_token_ids carried by each chunk, so a
    perf arm can still be validated against the reference without re-tokenizing.
    """

    stage: str
    started_at: str
    finished_at: str
    http_status: int | None
    error: str | None
    request_file: str
    response_file: str
    first_token_seconds: float | None = None
    total_seconds: float | None = None
    chunk_times: list[float] = field(default_factory=list)
    chunk_token_counts: list[int] = field(default_factory=list)
    completion_token_ids: list[int] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    cached_tokens: int | None = None
    worker: str | None = None
    prefill_worker: str | None = None
    usage: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return (
            self.http_status == 200
            and not self.error
            and self.first_token_seconds is not None
            and bool(self.completion_token_ids)
        )

    @property
    def tokens(self) -> int:
        return len(self.completion_token_ids)

    @property
    def tpot_seconds(self) -> float | None:
        """Mean time per output token after the first one."""
        if (
            self.first_token_seconds is None
            or self.total_seconds is None
            or self.tokens < 2
        ):
            return None
        return (self.total_seconds - self.first_token_seconds) / (self.tokens - 1)

    @property
    def inter_chunk_seconds(self) -> list[float]:
        return [b - a for a, b in zip(self.chunk_times, self.chunk_times[1:])]

    @property
    def tokens_per_second(self) -> float | None:
        if not self.total_seconds or self.tokens == 0:
            return None
        return self.tokens / self.total_seconds

    def sample(self) -> TokenSample:
        return TokenSample(
            tuple(self.prompt_token_ids),
            tuple(self.completion_token_ids),
            self.finish_reason,
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "ttft_seconds": self.first_token_seconds,
            "tpot_seconds": self.tpot_seconds,
            "total_seconds": self.total_seconds,
            "tokens": self.tokens,
            "tokens_per_second": self.tokens_per_second,
            "chunks": len(self.chunk_times),
            "cached_tokens": self.cached_tokens,
            "finish_reason": self.finish_reason,
            "worker": self.worker,
            "prefill_worker": self.prefill_worker,
        }


def _nvext(chunk: dict[str, Any]) -> dict[str, Any]:
    top = chunk.get("nvext")
    if isinstance(top, dict):
        return top
    choices = chunk.get("choices") or []
    if choices and isinstance(choices[0], dict):
        inner = choices[0].get("nvext")
        if isinstance(inner, dict):
            return inner
    return {}


class Client:
    def __init__(
        self,
        base_url: str,
        model: str,
        evidence: Path,
        *,
        max_tokens: int,
        request_timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.evidence = evidence
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.log: list[dict[str, Any]] = []

    def models(self, timeout: float) -> dict[str, Any]:
        with urlopen(self.base_url + "/v1/models", timeout=timeout) as response:
            return json.load(response)

    def completion(
        self,
        stage: str,
        prompt: str,
        *,
        worker: str | None = None,
        rank: int | None = None,
        prefill: str | None = None,
        max_tokens: int | None = None,
        extra_fields: list[str] | None = None,
    ) -> Response:
        headers = {"Content-Type": "application/json"}
        if worker is not None:
            headers["x-dynamo-worker-instance-id"] = str(worker)
        if prefill is not None:
            headers["x-dynamo-prefill-instance-id"] = str(prefill)
        if rank is not None:
            headers["x-dynamo-dp-rank"] = str(rank)
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": 0,
            "seed": 0,
            "nvext": {"extra_fields": extra_fields or EXTRA_FIELDS},
        }
        started_at = utc_now()
        request_file = f"{stage}.request.json"
        response_file = f"{stage}.response.json"
        save_json(
            self.evidence / request_file,
            {
                "started_at": started_at,
                "url": self.base_url + "/v1/completions",
                "headers": headers,
                "body": body,
            },
        )
        started = time.monotonic()
        status: int | None = None
        raw = b""
        parsed: dict[str, Any] | None = None
        error: str | None = None
        response_headers: dict[str, str] = {}
        try:
            request = Request(
                self.base_url + "/v1/completions",
                headers=headers,
                data=json.dumps(body).encode(),
                method="POST",
            )
            try:
                http = urlopen(request, timeout=self.request_timeout)
            except HTTPError as failure:
                http = failure
            with http:
                status = http.status
                response_headers = dict(http.headers.items())
                raw = http.read()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                parsed = None
                error = "response body is not a JSON object"
        except Exception as failure:  # noqa: BLE001 - evidence keeps every failure
            error = f"{type(failure).__name__}: {failure}"
        result = Response(
            stage=stage,
            started_at=started_at,
            finished_at=utc_now(),
            latency_seconds=time.monotonic() - started,
            http_status=status,
            body=parsed,
            error=error,
            request_file=request_file,
            response_file=response_file,
            headers=response_headers,
        )
        record = {
            **result.__dict__,
            "body_text": None if parsed is not None else raw.decode(errors="replace"),
        }
        save_json(self.evidence / response_file, record)
        self.log.append(
            {
                "stage": stage,
                "started_at": started_at,
                "finished_at": result.finished_at,
                "http_status": status,
                "request_file": request_file,
                "response_file": response_file,
            }
        )
        return result

    def stream_completion(
        self,
        stage: str,
        prompt: str,
        *,
        worker: str | None = None,
        max_tokens: int | None = None,
        extra_fields: list[str] | None = None,
        prefill: str | None = None,
    ) -> StreamResponse:
        """Stream one completion and time every chunk on the client side."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if worker is not None:
            headers["x-dynamo-worker-instance-id"] = str(worker)
        if prefill is not None:
            headers["x-dynamo-prefill-instance-id"] = str(prefill)
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "nvext": {"extra_fields": extra_fields or EXTRA_FIELDS + ["timing"]},
        }
        started_at = utc_now()
        request_file = f"{stage}.request.json"
        response_file = f"{stage}.stream.json"
        save_json(
            self.evidence / request_file,
            {
                "started_at": started_at,
                "url": self.base_url + "/v1/completions",
                "headers": headers,
                "body": body,
            },
        )
        result = StreamResponse(
            stage=stage,
            started_at=started_at,
            finished_at=started_at,
            http_status=None,
            error=None,
            request_file=request_file,
            response_file=response_file,
        )
        raw_chunks: list[dict[str, Any]] = []
        sent = time.monotonic()
        try:
            request = Request(
                self.base_url + "/v1/completions",
                headers=headers,
                data=json.dumps(body).encode(),
                method="POST",
            )
            try:
                http = urlopen(request, timeout=self.request_timeout)
            except HTTPError as failure:
                http = failure
            with http:
                result.http_status = http.status
                if http.status != 200:
                    result.error = http.read().decode(errors="replace")[:2000]
                else:
                    for line in http:
                        now = time.monotonic() - sent
                        text = line.decode(errors="replace").strip()
                        if not text.startswith("data:"):
                            continue
                        payload = text[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            result.error = f"undecodable SSE chunk: {payload[:120]}"
                            break
                        raw_chunks.append({"t": now, "chunk": chunk})
                        nvext = _nvext(chunk)
                        ids = nvext.get("completion_token_ids")
                        if isinstance(ids, list) and ids:
                            if result.first_token_seconds is None:
                                result.first_token_seconds = now
                            result.chunk_times.append(now)
                            result.chunk_token_counts.append(len(ids))
                            result.completion_token_ids.extend(int(i) for i in ids)
                        prompt_ids = nvext.get("prompt_token_ids")
                        if isinstance(prompt_ids, list) and prompt_ids and not result.prompt_token_ids:
                            result.prompt_token_ids = [int(i) for i in prompt_ids]
                        identity = nvext.get("worker_id") or {}
                        if isinstance(identity, dict) and identity.get("decode_worker_id"):
                            result.worker = str(identity["decode_worker_id"])
                        if isinstance(identity, dict) and identity.get("prefill_worker_id"):
                            result.prefill_worker = str(identity["prefill_worker_id"])
                        if isinstance(nvext.get("timing"), dict):
                            result.timing = nvext["timing"]
                        for choice in chunk.get("choices") or []:
                            if isinstance(choice, dict) and choice.get("finish_reason"):
                                result.finish_reason = choice["finish_reason"]
                        usage = chunk.get("usage")
                        if isinstance(usage, dict):
                            result.usage = usage
                            details = usage.get("prompt_tokens_details") or {}
                            if type(details.get("cached_tokens")) is int:
                                result.cached_tokens = details["cached_tokens"]
                    result.total_seconds = time.monotonic() - sent
                    if result.first_token_seconds is None and not result.error:
                        result.error = "stream carried no nvext.completion_token_ids"
        except Exception as failure:  # noqa: BLE001 - evidence keeps every failure
            result.error = f"{type(failure).__name__}: {failure}"
        result.finished_at = utc_now()
        save_json(
            self.evidence / response_file,
            {**{k: v for k, v in result.__dict__.items()}, "raw_chunks": raw_chunks},
        )
        self.log.append(
            {
                "stage": stage,
                "started_at": started_at,
                "finished_at": result.finished_at,
                "http_status": result.http_status,
                "request_file": request_file,
                "response_file": response_file,
                "stream": True,
            }
        )
        return result
