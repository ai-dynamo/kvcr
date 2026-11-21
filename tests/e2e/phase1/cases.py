# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Phase 1 test cases TC-01 .. TC-04 from the KVCR test plan v7.

Each case receives the running stack, the reference token samples, and an
evidence directory. A case records checks; the runner derives PASS/FAIL.
BlockedError marks the combination BLOCKED with the missing condition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from .client import Client, Response
from .common import (
    BLOCK_SIZE,
    BlockedError,
    CaseResult,
    TokenSample,
    ValidationError,
    cached_tokens,
    compare_tokens,
    extract_sample,
    utc_now,
)
from .stack import Stack
from .telemetry import (
    LOCAL_DELIVER,
    LOCAL_FILL,
    TIER_WRITE_BYTES,
    Window,
    read_window,
)


@dataclass(frozen=True)
class Prompts:
    """Deterministic prompt texts derived from one recorded run tag."""

    tag: str

    @staticmethod
    def section(tag: str, index: int) -> str:
        return f"{tag} section{index} cache routing data"

    def prefix(self, tag: str, start: int, stop: int) -> str:
        return " ".join(self.section(tag, i) for i in range(start, stop))

    @property
    def tail(self) -> str:
        return " Repeat the first word of this text exactly, then continue the list:"

    @property
    def tc01(self) -> str:
        return self.prefix(self.tag + "a", 0, 80) + self.tail

    def all(self) -> dict[str, str]:
        return {"tc01": self.tc01}


def parse_iso(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


class Context:
    """Everything a case needs; also the single place that talks to the stack."""

    def __init__(
        self,
        stack: Stack,
        client: Client,
        prompts: Prompts,
        references: dict[str, TokenSample],
        *,
        index_wait: float,
        telemetry_timeout: float,
        hit_block_tokens: int = BLOCK_SIZE,
    ) -> None:
        self.stack = stack
        self.client = client
        self.prompts = prompts
        self.references = references
        self.index_wait = index_wait
        self.telemetry_timeout = telemetry_timeout
        # Tokens per prefix-cache hit block. 64 (vLLM --block-size) for dense
        # models; hybrid models (Qwen3.5) raise the attention block to 576 so
        # cached_tokens is a multiple of that.
        self.hit_block_tokens = hit_block_tokens

    # -- request helpers ---------------------------------------------------

    def request(self, stage: str, prompt: str, worker_index: int) -> Response:
        self.stack.check_children()
        worker_id = self.stack.worker_ids[worker_index]
        del worker_id
        return self.client.completion(stage, prompt, **self.stack.pin(worker_index))

    def require_ok(
        self, result: CaseResult, response: Response, worker_index: int
    ) -> bool:
        ok = result.check(
            f"{response.stage}: HTTP 200 with JSON body",
            response.ok,
            response.error or f"status {response.http_status}",
        )
        if not ok:
            return False
        worker = self.stack.identity(response.body or {})
        return result.check(
            f"{response.stage}: served by worker {worker_index}",
            worker == self.stack.worker_ids[worker_index],
            f"{self.stack.served_role}_worker_id={worker}",
        )

    def compare(
        self, result: CaseResult, response: Response, reference_key: str
    ) -> bool:
        if not response.ok or response.body is None:
            return result.check(
                f"{response.stage}: tokens equal reference", False, "no body"
            )
        sample = extract_sample(response.body)
        problems = compare_tokens(self.references[reference_key], sample)
        return result.check(
            f"{response.stage}: tokens equal reference ({reference_key})",
            not problems,
            "; ".join(problems)
            or f"{sample.length} tokens, finish={sample.finish_reason}",
        )

    def cached(self, response: Response) -> int:
        value = cached_tokens(response.body or {})
        if value is None:
            raise ValidationError(f"{response.stage}: response lacks cached_tokens")
        return value

    def check_cold(self, result: CaseResult, response: Response) -> None:
        cached = self.cached(response)
        result.check(
            f"{response.stage}: cold request has 0 cached tokens",
            cached == 0,
            str(cached),
        )

    def check_hit(
        self, result: CaseResult, response: Response, *, max_blocks: int | None = None
    ) -> int:
        cached = self.cached(response)
        block = self.hit_block_tokens
        aligned, detail = hit_verdict(cached, block, max_blocks)
        result.check(
            f"{response.stage}: cache hit is positive and block-aligned",
            aligned,
            detail,
        )
        return cached // block

    # -- telemetry helpers -------------------------------------------------

    def wait_window(
        self,
        worker_index: int,
        start: str,
        predicate,
        *,
        tag: str = "",
    ) -> Window:
        """Poll a worker's log until the summed window satisfies predicate.

        The engine emits one interval sample per logging period; the sample
        that covers a request can arrive several seconds after the response.
        """
        started = parse_iso(start)
        deadline = time.monotonic() + self.telemetry_timeout
        log = self.stack.workers[worker_index].log_path
        result = read_window(log, started)
        while time.monotonic() < deadline:
            result = read_window(log, started)
            if predicate(result):
                break
            time.sleep(1)
        return result

    def settle(self, worker_index: int, start: str) -> Window:
        """Wait one extra interval so the window includes the request's sample."""
        started = parse_iso(start)
        log = self.stack.workers[worker_index].log_path
        deadline = time.monotonic() + self.telemetry_timeout
        while time.monotonic() < deadline:
            current = read_window(log, started)
            if current.samples >= 2:
                return current
            time.sleep(1)
        return read_window(log, started)

    def local_commit_checks(
        self, result: CaseResult, worker_index: int, start: str, label: str
    ):
        """The cold request's blocks were stored and committed into the KVCR tier."""
        window = self.wait_window(
            worker_index, start, lambda w: w.blocks(LOCAL_FILL) > 0
        )
        fills = window.blocks(LOCAL_FILL)
        result.check(
            f"{label}: KVCR local_fill committed blocks after cold request",
            fills > 0 and window.successes(LOCAL_FILL) >= 1,
            f"local_fill blocks={fills}, successes={window.successes(LOCAL_FILL)}, "
            f"tier write bytes={window.get(TIER_WRITE_BYTES)}",
        )
        result.check(
            f"{label}: no positive KVCR failed/partial metrics",
            not window.peer_failures,
            str(window.peer_failures),
        )
        result.measurements[label] = window.totals
        return window


def hit_verdict(
    cached: int, block: int, max_blocks: int | None = None
) -> tuple[bool, str]:
    """A served prefix counts as a hit only if cached_tokens is a positive
    multiple of the hit block (and within the bound when one is given). A
    correct answer with zero cached tokens is a recomputation, not a recovery.
    """
    aligned = cached > 0 and cached % block == 0
    detail = f"cached_tokens={cached} (block {block})"
    if max_blocks is not None:
        detail += f", upper bound {max_blocks * block}"
        aligned = aligned and cached <= max_blocks * block
    return aligned, detail


# -- cases -------------------------------------------------------------------


def tc01(ctx: Context) -> CaseResult:
    result = CaseResult("TC-01", "Integration generation correctness", "R1")
    prompt = ctx.prompts.tc01
    cold = ctx.request("tc01-cold-w0", prompt, 0)
    result.evidence += [cold.request_file, cold.response_file]
    if not ctx.require_ok(result, cold, 0):
        result.finish()
        return result
    ctx.compare(result, cold, "tc01")
    ctx.check_cold(result, cold)
    ctx.local_commit_checks(result, 0, cold.started_at, "tc01 store->commit")
    time.sleep(ctx.index_wait)
    reuse = ctx.request("tc01-reuse-w0", prompt, 0)
    result.evidence += [reuse.request_file, reuse.response_file]
    if ctx.require_ok(result, reuse, 0):
        ctx.compare(result, reuse, "tc01")
        ctx.check_hit(result, reuse)
        window = ctx.settle(0, reuse.started_at)
        result.check(
            "tc01-reuse-w0: no positive KVCR failed/partial metrics",
            not window.peer_failures,
            str(window.peer_failures),
        )
        result.measurements["reuse_window"] = window.totals
        result.measurements["reuse_kvcr_local_deliver_blocks"] = window.blocks(
            LOCAL_DELIVER
        )
    result.finish()
    return result


CASES = {"TC-01": tc01}


def run_case(name: str, ctx: Context) -> CaseResult:
    started = utc_now()
    try:
        result = CASES[name](ctx)
    except BlockedError as blocked:
        result = CaseResult(name, name, "")
        result.status = "BLOCKED"
        result.error = str(blocked)
        result.started_at = started
        result.finish()
    except Exception as error:  # noqa: BLE001 - every failure becomes evidence
        result = CaseResult(name, name, "")
        result.status = "FAIL"
        result.error = f"{type(error).__name__}: {error}"
        result.started_at = started
        result.finish()
    return result
