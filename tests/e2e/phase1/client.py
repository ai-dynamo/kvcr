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

from .common import save_json, utc_now

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
        max_tokens: int | None = None,
        extra_fields: list[str] | None = None,
    ) -> Response:
        headers = {"Content-Type": "application/json"}
        if worker is not None:
            headers["x-dynamo-worker-instance-id"] = str(worker)
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
