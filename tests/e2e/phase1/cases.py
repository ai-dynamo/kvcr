# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Phase 1 test cases TC-01 .. TC-04 from the KVCR test plan v7.

Each case receives the running stack, the reference token samples, and an
evidence directory. A case records checks; the runner derives PASS/FAIL.
BlockedError marks the combination BLOCKED with the missing condition.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .client import Client, Response
from .common import (
    BLOCK_SIZE,
    BlockedError,
    CaseResult,
    TokenSample,
    ValidationError,
    cached_tokens,
    compare_tokens,
    expected_hit_blocks,
    extract_sample,
    save_json,
    utc_now,
)
from .stack import GUARD, GUARD_PROMOTED, RECOVERY_DISABLED, Stack
from .telemetry import (
    LOCAL_DELIVER,
    LOCAL_FILL,
    REMOTE_DELIVER,
    SOURCE_WRITE,
    TIER_READ_BYTES,
    TIER_WRITE_BYTES,
    TRANSFER,
    Window,
    duration_count,
    latest_gauge,
    read_window,
)

PROMOTED = re.compile(
    r"KVCR_EVENT guard_promoted guard=(\d+) pool=(\S+) recovered_blocks=(\d+)"
)


@dataclass(frozen=True)
class Prompts:
    """Deterministic prompt texts derived from one recorded run tag."""

    tag: str
    # TC-02 prompt geometry in sections (~17 tokens each): the shared prefix P1
    # must hold at least two complete hit blocks and the full prompt at least
    # one more, so hybrid models whose attention block is 576 tokens need
    # 140 / 200 instead of the dense 40 / 80.
    tc02_shared: int = 40
    tc02_sections: int = 80
    # Serve the shared prefix P1 as its own request before the full prompt, so
    # an engine that only checkpoints SSM state at a request's last aligned
    # boundary (vLLM mamba_cache_mode=align) holds the state at the P1/P2
    # junction; otherwise a hybrid model cannot reuse a partial prefix at all.
    seed_prefix: bool = False
    # Extra sections Q shared by the full and partial prompts AFTER P1 but
    # never served on their own: the shared prefix then runs past P1's last
    # SSM checkpoint, to measure whether KVCR pulls attention pages beyond
    # the block the engine can actually reuse.
    tc02_extra_shared: int = 0

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

    @property
    def tc02_p1(self) -> str:
        return self.prefix(self.tag + "b", 0, self.tc02_shared)

    @property
    def tc02_q(self) -> str:
        """Shared continuation after P1 (empty unless tc02_extra_shared > 0)."""
        if not self.tc02_extra_shared:
            return ""
        end = self.tc02_shared + self.tc02_extra_shared
        return " " + self.prefix(self.tag + "q", self.tc02_shared, end)

    @property
    def tc02_full(self) -> str:
        start = self.tc02_shared + self.tc02_extra_shared
        rest = self.prefix(
            self.tag + "c", start, start + self.tc02_sections - self.tc02_shared
        )
        return self.tc02_p1 + self.tc02_q + " " + rest + self.tail

    @property
    def tc02_partial(self) -> str:
        start = self.tc02_shared + self.tc02_extra_shared
        rest = self.prefix(
            self.tag + "d", start, start + self.tc02_sections - self.tc02_shared
        )
        return self.tc02_p1 + self.tc02_q + " " + rest + self.tail

    @property
    def tc02_absent(self) -> str:
        return self.prefix(self.tag + "e", 0, 80) + self.tail

    @property
    def tc03(self) -> str:
        return self.prefix(self.tag + "f", 0, 80) + self.tail

    @property
    def tc03_isolated(self) -> str:
        return self.prefix(self.tag + "g", 0, 80) + self.tail

    @property
    def tc04(self) -> str:
        return self.prefix(self.tag + "h", 0, 80) + self.tail

    @property
    def tc04_fresh(self) -> str:
        return self.prefix(self.tag + "i", 0, 80) + self.tail

    @property
    def tc04_second(self) -> str:
        return self.prefix(self.tag + "j", 0, 80) + self.tail

    def short_filler(self, index: int) -> str:
        """~80 tokens: exactly one full 64-token block (cached, hence evicting)
        plus a tail. Sub-block prompts are freed, not cached, and evict nothing."""
        return self.prefix(f"{self.tag}y{index}", 0, 4) + self.tail

    def all(self) -> dict[str, str]:
        prompts = {
            name: getattr(self, name)
            for name in (
                "tc01",
                *(["tc02_p1"] if self.seed_prefix else []),
                "tc02_full",
                "tc02_partial",
                "tc02_absent",
                "tc03",
                "tc03_isolated",
                "tc04",
                "tc04_fresh",
                "tc04_second",
            )
        }
        return prompts


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
        recovery_timeout: float,
        hit_block_tokens: int = BLOCK_SIZE,
        engine_cached: dict[str, list[int | None]] | None = None,
        tc02_seed_prefix: bool = False,
        tc02_evict_target: int = 0,
        ssm_groups: int = 0,
        page_bytes: int = 0,
    ) -> None:
        self.stack = stack
        self.client = client
        self.prompts = prompts
        self.references = references
        self.index_wait = index_wait
        self.telemetry_timeout = telemetry_timeout
        self.recovery_timeout = recovery_timeout
        # Tokens per prefix-cache hit block. 64 (vLLM --block-size) for dense
        # models; hybrid models (Qwen3.5) raise the attention block to 576 so
        # cached_tokens is a multiple of that, and every KVCR transfer carries
        # extra SSM state pages, so delivered pages exceed hit blocks.
        self.hit_block_tokens = hit_block_tokens
        # Hybrid geometry: KVCR pages per hit = attention blocks + one state page
        # per SSM group (0 = dense, or unknown: only pages >= blocks is checked).
        self.ssm_groups = ssm_groups
        # Bytes per KVCR page when known: transfer bytes must equal pages * size.
        self.page_bytes = page_bytes
        # Reference-phase cached_tokens per prompt (KVCR disabled), request order.
        self.engine_cached = engine_cached or {}
        self.tc02_seed_prefix = tc02_seed_prefix
        self.tc02_evict_target = tc02_evict_target

    @property
    def engine_partial_prefix_reuse(self) -> bool | None:
        """Did the KVCR-disabled engine reuse the shared prefix of tc02_partial?

        The reference phase serves tc02_full then tc02_partial, so the first
        tc02_partial request is exactly TC-02's partial-prefix situation on the
        bare engine. None when the reference did not record it.
        """
        values = self.engine_cached.get("tc02_partial") or []
        if not values or values[0] is None:
            return None
        return values[0] > 0

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

    def delivery_covers(
        self, delivered: int | float, hit_blocks: int, *, hits: int = 1
    ) -> bool:
        """Delivered KVCR pages account for the reported hit.

        Dense: one page per 64-token block, so pages == hit blocks. Hybrid:
        attention pages plus one state page per SSM group and hit, so pages ==
        blocks + ssm_groups * hits when the geometry is given (--ssm-groups),
        else only pages >= blocks.
        """
        if hit_blocks <= 0:
            return False
        if self.hit_block_tokens == BLOCK_SIZE:
            return delivered == hit_blocks
        if self.ssm_groups > 0:
            return delivered == hit_blocks + self.ssm_groups * hits
        return delivered >= hit_blocks

    def expected_pages(self, hit_blocks: int, *, hits: int = 1) -> str:
        if self.hit_block_tokens == BLOCK_SIZE or self.ssm_groups <= 0:
            return str(hit_blocks)
        return f"{hit_blocks}+{self.ssm_groups}*{hits}"

    def bytes_match_pages(
        self, result: CaseResult, label: str, nbytes: int | float, pages: int | float
    ) -> None:
        """When the page size is known, transfer bytes must be pages * page size."""
        if self.page_bytes <= 0:
            return
        result.check(
            f"{label}: transfer bytes == pages * page bytes",
            pages > 0 and nbytes == pages * self.page_bytes,
            f"bytes={nbytes}, pages={pages}, page bytes={self.page_bytes}",
        )

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

    def peer_transfer_checks(
        self,
        result: CaseResult,
        *,
        source: int,
        target: int,
        start: str,
        hit_blocks: int,
        label: str,
        allow_partial: bool = False,
    ) -> None:
        """Source write and target delivery must agree with the reported hit.

        allow_partial accepts a delivery the target reports as 'partial' and the
        matching source pin miss for the uncovered key; both are the expected
        signals when the request shares only part of the seeded prefix.
        """
        source_window = self.wait_window(
            source,
            start,
            lambda w: w.blocks(SOURCE_WRITE) >= hit_blocks and hit_blocks > 0,
        )
        written_pages = max(hit_blocks, source_window.blocks(SOURCE_WRITE))
        target_window = self.wait_window(
            target,
            start,
            lambda w: w.blocks(REMOTE_DELIVER) >= written_pages and hit_blocks > 0,
        )
        delivered = target_window.blocks(REMOTE_DELIVER)
        written = source_window.blocks(SOURCE_WRITE)
        result.check(
            f"{label}: target remote_deliver pages cover the cached blocks",
            self.delivery_covers(delivered, hit_blocks),
            f"remote_deliver={delivered}, cached blocks={hit_blocks}, "
            f"expected pages={self.expected_pages(hit_blocks)}, "
            f"block tokens={self.hit_block_tokens}",
        )
        result.check(
            f"{label}: source source_write blocks == target remote_deliver blocks",
            written == delivered and written > 0,
            f"source_write={written}, remote_deliver={delivered}",
        )
        partial = target_window.get(duration_count(REMOTE_DELIVER, "partial"))
        delivered_ok = target_window.successes(REMOTE_DELIVER) + (
            partial if allow_partial else 0
        )
        result.check(
            f"{label}: transfer/source_write/remote_deliver successes recorded",
            source_window.successes(TRANSFER) >= 1
            and source_window.successes(SOURCE_WRITE) >= 1
            and delivered_ok >= 1,
            f"transfer={source_window.successes(TRANSFER)}, "
            f"source_write={source_window.successes(SOURCE_WRITE)}, "
            f"remote_deliver success={target_window.successes(REMOTE_DELIVER)} "
            f"partial={partial}",
        )
        source_bytes = source_window.bytes(SOURCE_WRITE)
        target_bytes = target_window.get(TIER_READ_BYTES)
        result.check(
            f"{label}: source bytes == target tier read bytes",
            source_bytes == target_bytes and source_bytes > 0,
            f"source_write bytes={source_bytes}, kvcr tier read bytes={target_bytes}",
        )
        self.bytes_match_pages(result, label, source_bytes, written)
        failures = source_window.peer_failures + target_window.peer_failures
        if allow_partial:
            expected = (
                duration_count(REMOTE_DELIVER, "partial"),
                duration_count("framework_pin_wait", "failed"),
            )
            failures = [
                entry
                for entry in failures
                if not any(
                    entry["metric"].startswith(name[: -len("_count")])
                    for name in expected
                )
            ]
        result.check(
            f"{label}: no unexpected KVCR failed/partial metrics",
            not failures,
            str(failures),
        )
        result.measurements[label] = {
            "source_window": source_window.totals,
            "target_window": target_window.totals,
        }

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

    def save(self, result: CaseResult, name: str, value: Any) -> None:
        path = self.client.evidence / f"{result.case}.{name}.json"
        save_json(path, value)
        result.evidence.append(path.name)


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


def tc02(ctx: Context) -> CaseResult:
    result = CaseResult("TC-02", "Local DRAM full flow", "R2")
    p = ctx.prompts
    if ctx.tc02_seed_prefix:
        prefix_req = ctx.request("tc02-seed-prefix-w0", p.tc02_p1, 0)
        result.evidence += [prefix_req.request_file, prefix_req.response_file]
        ctx.require_ok(result, prefix_req, 0)
        ctx.compare(result, prefix_req, "tc02_p1")
        ctx.check_cold(result, prefix_req)
        ctx.local_commit_checks(
            result, 0, prefix_req.started_at, "tc02 prefix store->commit on w0"
        )
        time.sleep(ctx.index_wait)
    seed = ctx.request("tc02-seed-full-w0", p.tc02_full, 0)
    result.evidence += [seed.request_file, seed.response_file]
    if not ctx.require_ok(result, seed, 0):
        result.finish()
        return result
    ctx.compare(result, seed, "tc02_full")
    if ctx.tc02_seed_prefix:
        # P1 was served first (below, before this request), so the full prompt
        # is itself a partial-prefix hit on worker 0.
        seed_hit_blocks = ctx.check_hit(result, seed)
        result.measurements["tc02_seed_full_hit_blocks"] = seed_hit_blocks
    else:
        ctx.check_cold(result, seed)
    ctx.local_commit_checks(result, 0, seed.started_at, "tc02 store->commit on w0")
    time.sleep(ctx.index_wait)

    # partial hit: worker 1 is cold; only the shared P1 blocks can be delivered.
    partial = ctx.request("tc02-partial-w1", p.tc02_partial, 1)
    result.evidence += [partial.request_file, partial.response_file]
    if ctx.require_ok(result, partial, 1):
        ctx.compare(result, partial, "tc02_partial")
        full_ids = extract_sample(seed.body or {}).prompt_token_ids
        part_ids = extract_sample(partial.body or {}).prompt_token_ids
        shared = 0
        for left, right in zip(full_ids, part_ids):
            if left != right:
                break
            shared += 1
        bound = expected_hit_blocks(len(part_ids), shared, ctx.hit_block_tokens)
        result.measurements["tc02_partial_shared_prefix_tokens"] = shared
        result.measurements["tc02_prompt_tokens"] = {
            "full": len(full_ids),
            "partial": len(part_ids),
        }
        result.measurements["engine_partial_prefix_reuse"] = {
            "reference_cached_tokens": ctx.engine_cached.get("tc02_partial"),
            "reuses": ctx.engine_partial_prefix_reuse,
        }
        seeded_blocks = (len(full_ids) - 1) // ctx.hit_block_tokens
        if ctx.engine_partial_prefix_reuse is False:
            # The bare engine (KVCR disabled) reported cached_tokens=0 for this
            # very prompt right after serving the seeded one: this vLLM build
            # does not reuse a partial prefix for the model (seen with Qwen3.5
            # hybrid attention, SSM state only valid for an exact prompt), so a
            # KVCR hit cannot exist either. Record what KVCR did instead of
            # failing the case on an engine limit.
            window = ctx.settle(1, partial.started_at)
            source = ctx.settle(0, partial.started_at)
            result.check(
                "tc02-partial-w1: partial-prefix reuse unsupported by the engine "
                "itself (reference cached_tokens=0), hit checks not applicable",
                True,
                f"cached_tokens={ctx.cached(partial)}, remote_deliver pages="
                f"{window.blocks(REMOTE_DELIVER)}, source_write pages="
                f"{source.blocks(SOURCE_WRITE)}, target "
                f"failures={window.peer_failures}",
            )
            result.measurements["tc02 partial (engine limit)"] = {
                "target_window": window.totals,
                "source_window": source.totals,
            }
        else:
            blocks = ctx.check_hit(result, partial, max_blocks=bound)
            result.check(
                "tc02-partial-w1: hit covers fewer blocks than the seeded prompt",
                blocks < seeded_blocks,
                f"hit blocks={blocks}, seeded prompt blocks={seeded_blocks}",
            )
            ctx.peer_transfer_checks(
                result,
                source=0,
                target=1,
                start=partial.started_at,
                hit_blocks=blocks,
                label="tc02 partial deliver",
                allow_partial=True,
            )

    if ctx.require_ok(result, partial, 1):
        # Diagnostic (2026-10-08, hybrid): when the partial request's tokens
        # diverge from the reference, tell numerics apart from a corrupted
        # delivery by repeating the identical prompt on worker 1 (now a pure
        # vLLM GPU prefix-cache hit) and on worker 0 (KVCR-local, shares the
        # seeded prefix). Same tokens everywhere = numerics of this prompt;
        # only the first answer differing = the delivery path.
        first = extract_sample(partial.body or {})
        for stage, worker, key in (
            ("tc02-partial-again-w1", 1, "tc02_partial"),
            ("tc02-partial-w0", 0, "tc02_partial"),
        ):
            again = ctx.request(stage, p.tc02_partial, worker)
            result.evidence += [again.request_file, again.response_file]
            if ctx.require_ok(result, again, worker):
                ctx.compare(result, again, key)
                same = compare_tokens(first, extract_sample(again.body or {}))
                same = [x for x in same if "prompt token IDs" not in x]
                result.check(
                    f"{stage}: tokens equal the first partial answer on worker 1",
                    not same,
                    "; ".join(same) or f"identical, cached_tokens={ctx.cached(again)}",
                )
                result.measurements[f"{stage}_cached_tokens"] = ctx.cached(again)

    # Diagnostic (2026-10-11): after the delivery, evict worker 1's GPU prefix
    # cache with many sub-block prompts (one GPU block each, never a full block,
    # so nothing is offloaded to the host tier) and request the partial prompt
    # again on worker 1. Tells where the delivered prefix now lives: a hit with
    # remote_deliver=0 and host-tier load bytes = served from worker 1's own
    # vLLM DRAM tier; a hit with remote_deliver>0 = pulled from worker 0 again;
    # cached_tokens=0 = gone.
    if ctx.tc02_evict_target > 0 and ctx.require_ok(result, partial, 1):
        evict_start = utc_now()
        fillers = [
            (f"short_{i}", p.short_filler(i)) for i in range(ctx.tc02_evict_target)
        ]
        pin = ctx.stack.pin(1)

        def one(item: tuple[str, str]) -> Response:
            name, prompt = item
            return ctx.client.completion(
                f"tc02-evict-{name}", prompt, max_tokens=1, **pin
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            evict_responses = list(pool.map(one, fillers))
        result.check(
            "tc02 evict: every short filler request succeeded on worker 1",
            all(r.ok for r in evict_responses),
            f"{sum(r.ok for r in evict_responses)} of {len(evict_responses)} ok",
        )
        evict_window = ctx.settle(1, evict_start)
        time.sleep(ctx.index_wait)
        again2 = ctx.request("tc02-partial-after-evict-w1", p.tc02_partial, 1)
        result.evidence += [again2.request_file, again2.response_file]
        if ctx.require_ok(result, again2, 1):
            ctx.compare(result, again2, "tc02_partial")
            window2 = ctx.settle(1, again2.started_at)
            cached2 = ctx.cached(again2)
            served_from = (
                "gone (cold)"
                if cached2 == 0
                else "worker 0 via KVCR again"
                if window2.blocks(REMOTE_DELIVER) > 0
                else "worker 1 local (host tier or GPU)"
            )
            result.check(
                "tc02 after-evict: delivered prefix location recorded (informational)",
                True,
                f"cached_tokens={cached2}, "
                f"remote_deliver={window2.blocks(REMOTE_DELIVER)}, "
                f"local_deliver={window2.blocks(LOCAL_DELIVER)}, "
                f"host load bytes={window2.get('vllm:kv_offload_load_bytes')}, -> "
                f"{served_from}",
            )
            result.measurements["tc02_after_evict"] = {
                "cached_tokens": cached2,
                "served_from": served_from,
                "filler_requests": len(fillers),
                "filler_window": evict_window.totals,
                "request_window": window2.totals,
            }

    # present: the remainder (P2) still has to come from worker 0's KVCR pool.
    present = ctx.request("tc02-present-w1", p.tc02_full, 1)
    result.evidence += [present.request_file, present.response_file]
    if ctx.require_ok(result, present, 1):
        ctx.compare(result, present, "tc02_full")
        blocks = ctx.check_hit(result, present)
        window = ctx.wait_window(
            1, present.started_at, lambda w: w.blocks(REMOTE_DELIVER) > 0
        )
        result.check(
            "tc02-present-w1: remaining blocks delivered from worker 0 via KVCR",
            window.blocks(REMOTE_DELIVER) > 0 and window.successes(REMOTE_DELIVER) >= 1,
            f"remote_deliver blocks={window.blocks(REMOTE_DELIVER)}, "
            f"hit blocks={blocks}",
        )
        result.check(
            "tc02-present-w1: no positive KVCR failed/partial metrics",
            not window.peer_failures,
            str(window.peer_failures),
        )
        result.measurements["tc02 present deliver"] = window.totals

    # absent: never-seen data must not be reported as a hit anywhere.
    absent = ctx.request("tc02-absent-w1", p.tc02_absent, 1)
    result.evidence += [absent.request_file, absent.response_file]
    if ctx.require_ok(result, absent, 1):
        ctx.compare(result, absent, "tc02_absent")
        ctx.check_cold(result, absent)
        window = ctx.settle(1, absent.started_at)
        result.check(
            "tc02-absent-w1: no remote delivery for a miss",
            window.blocks(REMOTE_DELIVER) == 0,
            f"remote_deliver blocks={window.blocks(REMOTE_DELIVER)}",
        )

    # release: consumers finished; the source still owns its committed blocks.
    settle = ctx.settle(0, absent.started_at)
    allocated = latest_gauge(ctx.stack.workers[0].log_path, "local_g2_allocated_slots")
    pinned = latest_gauge(ctx.stack.workers[0].log_path, "pinned_keys")
    result.check(
        "tc02 release: source pool still holds committed blocks after delivery",
        allocated is not None and allocated > 0,
        f"local_g2_allocated_slots={allocated}",
    )
    result.check(
        "tc02 release: source framework pins released after delivery",
        pinned == 0,
        f"pinned_keys={pinned}",
    )
    result.measurements["release_window"] = settle.totals
    result.finish()
    return result


def tc03(ctx: Context) -> CaseResult:
    result = CaseResult("TC-03", "Request hints and peer transfer, same-node", "R4")
    p = ctx.prompts
    seed = ctx.request("tc03-seed-w0", p.tc03, 0)
    result.evidence += [seed.request_file, seed.response_file]
    if not ctx.require_ok(result, seed, 0):
        result.finish()
        return result
    ctx.compare(result, seed, "tc03")
    ctx.check_cold(result, seed)
    ctx.local_commit_checks(result, 0, seed.started_at, "tc03 source commits prefix")
    before = read_window(ctx.stack.workers[1].log_path, parse_iso(seed.started_at))
    result.check(
        "tc03: target worker 1 is cold for this prefix",
        before.blocks(REMOTE_DELIVER) == 0,
        f"remote_deliver before request={before.blocks(REMOTE_DELIVER)}",
    )
    time.sleep(ctx.index_wait)

    # Two requests reach worker 1 together: one hinted (seeded on worker 0) and
    # one fresh. Hints are request-scoped, so the fresh one must not benefit.
    with ThreadPoolExecutor(max_workers=2) as pool:
        hinted_future = pool.submit(ctx.request, "tc03-hinted-w1", p.tc03, 1)
        isolated_future = pool.submit(
            ctx.request, "tc03-isolated-w1", p.tc03_isolated, 1
        )
        hinted = hinted_future.result()
        isolated = isolated_future.result()
    result.evidence += [
        hinted.request_file,
        hinted.response_file,
        isolated.request_file,
        isolated.response_file,
    ]
    blocks = 0
    if ctx.require_ok(result, hinted, 1):
        ctx.compare(result, hinted, "tc03")
        blocks = ctx.check_hit(result, hinted)
    if ctx.require_ok(result, isolated, 1):
        ctx.compare(result, isolated, "tc03_isolated")
        ctx.check_cold(result, isolated)
    start = min(hinted.started_at, isolated.started_at)
    if blocks:
        ctx.peer_transfer_checks(
            result,
            source=0,
            target=1,
            start=start,
            hit_blocks=blocks,
            label="tc03 transfer",
        )
    result.finish()
    return result


def tc04(ctx: Context) -> CaseResult:
    result = CaseResult("TC-04", "Guard protection, takeover, re-attach", "R3")
    if ctx.stack.settings.mode != GUARD:
        raise BlockedError("TC-04 runs only in Guard mode")
    p = ctx.prompts
    service = ctx.stack.services[0]
    seed = ctx.request("tc04-protect-w0", p.tc04, 0)
    result.evidence += [seed.request_file, seed.response_file]
    if not ctx.require_ok(result, seed, 0):
        result.finish()
        return result
    ctx.compare(result, seed, "tc04")
    ctx.check_cold(result, seed)
    committed = ctx.local_commit_checks(
        result, 0, seed.started_at, "tc04 protected commit"
    )
    committed_blocks = committed.blocks(LOCAL_FILL)
    second_seed = ctx.request("tc04-protect-second-w0", p.tc04_second, 0)
    result.evidence += [second_seed.request_file, second_seed.response_file]
    if ctx.require_ok(result, second_seed, 0):
        ctx.compare(result, second_seed, "tc04_second")
        ctx.check_cold(result, second_seed)
        second = ctx.local_commit_checks(
            result, 0, second_seed.started_at, "tc04 protected commit (second prompt)"
        )
        committed_blocks += second.blocks(LOCAL_FILL)
    result.check(
        "tc04: warm recovery still enabled before failure",
        RECOVERY_DISABLED not in service.log_text()
        and RECOVERY_DISABLED not in ctx.stack.workers[0].log_text(),
        "no 'KVCR pool recovery disabled' in service or worker log",
    )
    time.sleep(ctx.index_wait)

    # Engine failure: the whole worker process group dies; its service survives.
    promotions_before = service.log_text().count(GUARD_PROMOTED)
    kill = ctx.stack.kill_worker(0)
    ctx.save(result, "kill", kill)
    killed = time.monotonic()
    text = ctx.stack.wait_log(
        service, GUARD_PROMOTED, ctx.recovery_timeout, promotions_before + 1
    )
    promoted_after = time.monotonic() - killed
    match = PROMOTED.findall(text)[-1] if PROMOTED.findall(text) else None
    recovered = int(match[2]) if match else -1
    result.measurements["guard_promotion_seconds_after_kill"] = round(promoted_after, 3)
    result.measurements["guard_recovered_blocks"] = recovered
    result.measurements["committed_blocks_before_failure"] = committed_blocks
    result.check(
        "tc04: service kept running and promoted its Guard after engine death",
        service.alive() and match is not None,
        f"guard_promoted after {promoted_after:.1f}s: {match}",
    )
    result.check(
        "tc04: Guard recovered the committed blocks",
        recovered >= committed_blocks > 0,
        f"recovered_blocks={recovered}, committed local_fill blocks={committed_blocks}",
    )

    # Takeover: worker 1 reads preserved prefixes from the Guard. Worker 1 had
    # already exchanged NIXL metadata with the dead primary at this endpoint, so
    # the first and second requests are checked separately.
    for attempt, (stage, key) in enumerate(
        (
            ("tc04-guard-served-w1", "tc04"),
            ("tc04-guard-served-second-w1", "tc04_second"),
        ),
        start=1,
    ):
        served = ctx.request(stage, getattr(p, key), 1)
        result.evidence += [served.request_file, served.response_file]
        result.measurements[f"guard_served_request_{attempt}_latency_seconds"] = round(
            served.latency_seconds, 3
        )
        if not ctx.require_ok(result, served, 1):
            continue
        ctx.compare(result, served, key)
        blocks = ctx.check_hit(result, served)
        window = ctx.wait_window(
            1,
            served.started_at,
            lambda w, b=blocks: w.blocks(REMOTE_DELIVER) >= b and b > 0,
        )
        result.check(
            f"{stage}: Guard-served blocks delivered to worker 1 (remote_deliver)",
            ctx.delivery_covers(window.blocks(REMOTE_DELIVER), blocks)
            and window.successes(REMOTE_DELIVER) >= 1,
            f"remote_deliver blocks={window.blocks(REMOTE_DELIVER)}, "
            f"hit blocks={blocks}, expected pages={ctx.expected_pages(blocks)}, "
            f"tier read bytes={window.get(TIER_READ_BYTES)}",
        )
        ctx.bytes_match_pages(
            result, stage, window.get(TIER_READ_BYTES), window.blocks(REMOTE_DELIVER)
        )
        result.check(
            f"{stage}: no positive KVCR failed/partial metrics while Guard served",
            not window.peer_failures,
            str(window.peer_failures),
        )
        result.measurements[f"guard_served_window_{attempt}"] = window.totals
        result.measurements[f"guard_service_log_after_request_{attempt}"] = [
            line[:300]
            for line in service.log_text().splitlines()
            if "WARNING" in line or "Error" in line
        ][-6:]

    # Re-attach: a replacement worker claims the pool group back.
    ctx.stack.start_worker(0, tag="-replacement")
    reattached = time.monotonic()
    ctx.stack.wait_model(ctx.client)
    ctx.stack.discover_worker(ctx.client, 0)
    result.measurements["replacement_ready_seconds"] = round(
        time.monotonic() - reattached, 3
    )
    replacement = ctx.stack.workers[0]
    result.check(
        "tc04: replacement attached without a second promotion (single owner)",
        service.alive()
        and service.log_text().count(GUARD_PROMOTED) == promotions_before + 1
        and RECOVERY_DISABLED not in replacement.log_text(),
        f"guard_promoted events={service.log_text().count(GUARD_PROMOTED)}",
    )
    fresh = ctx.request("tc04-fresh-w0-replacement", p.tc04_fresh, 0)
    result.evidence += [fresh.request_file, fresh.response_file]
    if ctx.require_ok(result, fresh, 0):
        ctx.compare(result, fresh, "tc04_fresh")
        ctx.check_cold(result, fresh)
        ctx.local_commit_checks(
            result, 0, fresh.started_at, "tc04 replacement commits new data"
        )
    again = ctx.request("tc04-preserved-w0-replacement", p.tc04, 0)
    result.evidence += [again.request_file, again.response_file]
    if ctx.require_ok(result, again, 0):
        ctx.compare(result, again, "tc04")
        # The preserved prefix must be served from the reclaimed pool, not
        # recomputed: a positive, block-aligned hit with no KVCR failures.
        ctx.check_hit(result, again)
        window = ctx.settle(0, again.started_at)
        result.check(
            "tc04-preserved-w0-replacement: no positive KVCR failed/partial metrics",
            not window.peer_failures,
            str(window.peer_failures),
        )
        result.measurements["replacement_preserved_prefix_cached_tokens"] = ctx.cached(
            again
        )
        result.measurements["replacement_preserved_prefix_window"] = window.totals
    result.finish()
    return result


CASES = {"TC-01": tc01, "TC-02": tc02, "TC-03": tc03, "TC-04": tc04}


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
