# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run KVCR test plan v7 Phase 1 (TC-01 -> TC-06) on one node with two GPUs.

Sequence:
  1. preflight        GPUs, shared memory, kernel (Guard), ports
  2. reference        KVCR disabled worker; every prompt generated twice;
                      perf prompts are also streamed cold + warm for the
                      KVCR-disabled TTFT/TPOT baseline (TC-05)
  3. stack            per-worker kvcr_service (Guard) + KV router + 2 workers
  4. cases            TC-01, TC-02, TC-03 same-node, TC-04 (Guard only),
                      TC-05 perf (cold/local/remote), TC-06 Guard recovery timing
  5. evidence         result.json, junit.xml, all requests/responses/logs

Exit status is 0 only when every selected case is PASS. BLOCKED cases are
reported as skipped with the missing condition and make the run fail unless
--allow-blocked is given.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import os
import re
import socket
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1.cases import CASES, Context, Prompts, run_case  # noqa: E402
from phase1.client import Client  # noqa: E402
from phase1.perf import PERF_CASES, STREAM_FIELDS  # noqa: E402
from phase1.common import (  # noqa: E402
    BLOCKED,
    FAIL,
    PASS,
    BlockedError,
    TokenSample,
    ValidationError,
    compare_tokens,
    extract_sample,
    save_json,
    utc_now,
    write_junit,
)
from phase1.stack import GUARD, MODES, Settings, Stack  # noqa: E402

CASES.update(PERF_CASES)
PERF_CASE_NAMES = set(PERF_CASES)


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    versions = {}
    for distribution in (
        "kvcr",
        "nixl",
        "vllm",
        "ai-dynamo",
        "ai-dynamo-runtime",
        "torch",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    manifest = Path("/opt/kvcr-build/source-revisions.json")
    package = Path(__file__).resolve().parent / "phase1"
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "image_source_revisions": json.loads(manifest.read_text())
        if manifest.exists()
        else None,
        "harness_sha256": digest.hexdigest(),
    }


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output_dir.resolve()
        if self.output.exists():
            # A remote kvcr_service sidecar (two-node run) logs into
            # <output>/stack before the harness starts; anything else is stale.
            stray = [
                p for p in self.output.rglob("*")
                if p.is_file() and not (
                    p.parent == self.output / "stack"
                    and re.fullmatch(r"kvcr-service-\d\.log", p.name)
                )
            ]
            if stray:
                raise ValidationError(
                    f"output directory must be empty: {self.output} (found {stray[0]})"
                )
        self.output.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + args.timeout
        self.settings = Settings(
            model=args.model,
            output=self.output,
            mode=args.mode,
            topology=args.topology,
            gpus=tuple(args.gpus),
            http_port=args.http_port,
            reference_http_port=args.reference_http_port,
            pool_gb=args.pool_gb,
            compatibility_digest=args.compatibility_digest or "",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            batch_invariant=args.batch_invariant,
            state_agent=not args.no_state_agent,
            readiness_timeout=args.readiness_timeout,
            request_timeout=args.request_timeout,
            extra_worker_args=list(args.worker_arg),
            frontend_env=dict(item.split("=", 1) for item in args.frontend_env if "=" in item),
            disable_hybrid_kv_cache_manager=not args.keep_hybrid_kv_cache_manager,
            enforce_eager=not args.no_enforce_eager,
            external_services=args.external_services,
            remote_workers=args.remote_workers or None,
            etcd_endpoints=args.etcd,
        )
        self.stack = Stack(self.settings, self.output / "stack")
        self.stack.deadline_check = self.check_deadline
        self.tag = args.prompt_tag or uuid.uuid4().hex[:12]
        self.prompts = Prompts(
            self.tag,
            perf_count=args.perf_repeats,
            tc02_shared=args.tc02_shared_sections,
            tc02_sections=args.tc02_sections,
            seed_prefix=args.tc02_seed_prefix,
        )
        self.references: dict[str, TokenSample] = {}
        self.baseline_timings: dict[str, Any] = {}
        # cached_tokens the KVCR-disabled engine reported per reference request,
        # in order; tells a case what the engine alone can reuse.
        self.reference_cached: dict[str, list[int | None]] = {}
        self.wants_perf = bool(PERF_CASE_NAMES & set(args.cases))
        self.result: dict[str, Any] = {
            "plan": "KVCR Test Plan v7, Phase 1 (Guard-mode critical path)",
            "started_at": utc_now(),
            "status": "running",
            "mode": args.mode,
            "deployment": "aggregated",
            "model": args.model,
            "prompt_tag": self.tag,
            "max_tokens": args.max_tokens,
            "perf": {
                "repeats": args.perf_repeats,
                "max_tokens": args.perf_max_tokens,
                "concurrency": args.perf_concurrency,
                "evict_filler": args.perf_evict_filler,
                "concurrency_levels": args.perf_concurrency_levels,
                "hit_block_tokens": args.hit_block_tokens,
                "tc02_sections": [args.tc02_shared_sections, args.tc02_sections],
                "tc02_seed_prefix": args.tc02_seed_prefix,
            },
            "worker_args": list(args.worker_arg),
            "disable_hybrid_kv_cache_manager": not args.keep_hybrid_kv_cache_manager,
            "enforce_eager": not args.no_enforce_eager,
            "topology": ("two-node" if args.remote_workers
                         else "sidecar" if args.external_services else "single-container"),
            "remote_workers": args.remote_workers or None,
            "discovery": "etcd" if args.etcd else "file",
            "external_services": str(args.external_services) if args.external_services else None,
            "batch_invariant": args.batch_invariant,
            "state_agent": not args.no_state_agent,
            "selected_cases": list(args.cases),
            "cases": [],
            "reference": {},
            "limitations": [
                "Single node, two GPUs, aggregated deployment only; disaggregated "
                "1P+1D has no valid KVCR source/target here and is not run.",
                "Local-hit path evidence is the KVCR commit (local_fill) after the "
                "cold "
                "request plus the reuse hit count; remote and Guard paths have "
                "transfer telemetry as direct evidence.",
            ],
        }
        self.save()

    def save(self) -> None:
        save_json(self.output / "result.json", self.result)

    def check_deadline(self) -> None:
        if time.monotonic() > self.deadline:
            raise ValidationError("overall run deadline expired")

    # -- phases ------------------------------------------------------------

    def reference_phase(self) -> None:
        """Establish reference tokens from a KVCR-disabled engine, twice each."""
        evidence = self.output / "reference"
        evidence.mkdir()
        stack = self.stack
        stack.start_frontend(
            self.settings.reference_http_port, name="reference-frontend"
        )
        worker = stack.start_worker(0, kvcr=False, tag="-reference")
        decode_ref = (
            stack.start_worker(2, kvcr=False, tag="-reference") if stack.disagg else None
        )
        client = Client(
            f"http://127.0.0.1:{self.settings.reference_http_port}",
            self.args.model,
            evidence,
            max_tokens=self.args.max_tokens,
            request_timeout=self.args.request_timeout,
        )
        stack.wait_model(client)
        if stack.disagg:
            stack.discover_worker(client, 2)
        stack.discover_worker(client, 0)
        launch = json.loads((stack.evidence / f"{worker.name}.launch.json").read_text())
        bypass = not any(
            "OffloadingConnector" in part or '"kvcr"' in part for part in launch["command"]
        )
        samples: dict[str, list[TokenSample]] = {}
        unstable = []
        unstable_detail: dict[str, Any] = {}
        perf_names = set(self.prompts.perf_names()) if self.wants_perf else set()
        # TC-06 prompts are generated with the perf token budget too, but need no
        # KVCR-disabled timing baseline.
        long_names = perf_names | (set(self.prompts.tc06_names()) if self.wants_perf else set())
        # Only the prompts the requested cases use: a long prompt of a case not
        # being run must not block the run when the bare engine is unstable on it.
        case_prompts = {
            "TC-01": {"tc01"},
            "TC-02": {"tc02_p1", "tc02_full", "tc02_partial", "tc02_absent"},
            "TC-03": {"tc03", "tc03_isolated"},
            "TC-04": {"tc04", "tc04_fresh", "tc04_second"},
            "TC-05": set(perf_names),
            "TC-06": set(self.prompts.tc06_names()) | set(perf_names),
        }
        wanted: set[str] = set()
        for case in self.args.cases:
            wanted |= case_prompts.get(case, set())
        selected = {
            name: prompt for name, prompt in self.prompts.all().items() if name in wanted
        }
        for name, prompt in selected.items():
            runs = []
            if name in perf_names:
                # First contact with this prompt: KVCR-disabled cold TTFT/TPOT.
                # Second stream: vLLM's own prefix-cache hit, the natural
                # comparison for KVCR's local hit. Both must reproduce tokens.
                timings: dict[str, Any] = {}
                for kind in ("cold", "warm"):
                    streamed = client.stream_completion(
                        f"ref-{name}-{kind}",
                        prompt,
                        **stack.pin(0),
                        max_tokens=self.args.perf_max_tokens,
                        extra_fields=STREAM_FIELDS,
                    )
                    if not streamed.ok:
                        raise ValidationError(
                            f"reference stream {name} ({kind}) failed: {streamed.error}"
                        )
                    timings[kind] = streamed.metrics()
                    runs.append(streamed.sample())
                self.baseline_timings[name] = timings
            for repeat in (1, 2):
                response = client.completion(
                    f"ref-{name}-{repeat}",
                    prompt,
                    **stack.pin(0),
                    max_tokens=self.args.perf_max_tokens if name in long_names else None,
                )
                if not response.ok or response.body is None:
                    raise ValidationError(
                        f"reference request {name} failed: {response.error}"
                    )
                runs.append(extract_sample(response.body))
                details = (response.body.get("usage") or {}).get("prompt_tokens_details") or {}
                cached_value = details.get("cached_tokens")
                self.reference_cached.setdefault(name, []).append(
                    cached_value if type(cached_value) is int else None
                )
            samples[name] = runs
            # The non-streaming runs carry prompt IDs; streamed runs may not.
            canonical = runs[-1]
            labels = (["cold", "warm"] if name in perf_names else []) + ["1", "2"]
            divergence = {}
            for label, run in zip(labels, runs):
                problems = [
                    p for p in compare_tokens(canonical, run) if "prompt token IDs" not in p
                ]
                if problems:
                    first = next(
                        (
                            i
                            for i, (x, y) in enumerate(
                                zip(canonical.completion_token_ids, run.completion_token_ids)
                            )
                            if x != y
                        ),
                        min(canonical.length, run.length),
                    )
                    divergence[label] = {"first_divergent_token": first, "problems": problems}
            if divergence:
                unstable.append(name)
                unstable_detail[name] = divergence
            else:
                self.references[name] = canonical
        # Filesystem paths the harness itself chose may contain "kvcr" (an HF
        # cache under /cache/doxu-kvcr/, the evidence tree); they say nothing
        # about the engine, so blank them before counting mentions.
        own_names = [str(self.output), os.environ.get("HF_HOME", ""), socket.gethostname(),
                     self.settings.etcd_endpoints or "", *(self.settings.remote_workers or {}).values()]
        scrubbed = worker.log_text()
        for name in sorted({n for n in own_names if n and "kvcr" in n.lower()}, key=len, reverse=True):
            scrubbed = scrubbed.replace(name, "<harness-name>")
        kvcr_mentions = len(re.findall(r"(?i)kvcr", scrubbed))
        kvcr_lines = [
            line[:200]
            for line in scrubbed.splitlines()
            if re.search(r"(?i)kvcr", line)
        ][:10]
        self.result["reference"] = {
            "kvcr_disabled_in_launch": bypass,
            "batch_invariant": self.settings.batch_invariant,
            "unstable": unstable_detail,
            "kvcr_mentions_in_worker_log": kvcr_mentions,
            "cached_tokens": self.reference_cached,
            "kvcr_lines_in_worker_log": kvcr_lines,
            "prompts": {
                name: {
                    "prompt_tokens": max(len(run.prompt_token_ids) for run in runs),
                    "completion_tokens": [run.length for run in runs],
                    "finish_reason": [run.finish_reason for run in runs],
                    "stable": name not in unstable,
                }
                for name, runs in samples.items()
            },
        }
        save_json(
            evidence / "reference-tokens.json",
            {name: sample.__dict__ for name, sample in self.references.items()},
        )
        if self.baseline_timings:
            self.result["reference"]["kvcr_disabled_timings"] = self.baseline_timings
            save_json(evidence / "baseline-timings.json", self.baseline_timings)
        self.save()
        stack.stop(*[c for c in (worker, decode_ref, stack.frontend) if c is not None])  # type: ignore[arg-type]
        # Forget the reference handles: the cases phase may start the decode
        # worker before worker 0 and must not see the stopped reference worker.
        for index in (0, 2):
            stack.workers.pop(index, None)
            stack.worker_ids.pop(index, None)
        stack.reset_discovery()
        if not bypass or kvcr_mentions:
            raise ValidationError("reference engine was not verified KVCR-free")
        if unstable:
            raise ValidationError(f"reference generation unstable for: {unstable}")

    def stack_phase(self) -> Client:
        stack = self.stack
        client = Client(
            f"http://127.0.0.1:{self.settings.http_port}",
            self.args.model,
            self.output / "requests",
            max_tokens=self.args.max_tokens,
            request_timeout=self.args.request_timeout,
        )
        client.evidence.mkdir()
        if self.settings.mode == GUARD:
            for index in (0, 1):
                stack.start_service(index)
            if stack.uses_state_agent():
                stack.start_state_agent()
        stack.start_frontend(self.settings.http_port)
        if stack.disagg:
            # The frontend lists the model only once a prefill worker is
            # registered, so bring up prefill 0 and the decode worker together
            # and identify the decode worker first.
            stack.start_worker(0)
            stack.start_worker(2)
            stack.wait_model(client)
            stack.discover_worker(client, 2)
            stack.discover_worker(client, 0)
            remaining = (1,)
        else:
            remaining = (0, 1)
        for index in remaining:
            stack.start_worker(index)
            stack.wait_model(client)
            stack.discover_worker(client, index)
        self.result["workers"] = {
            str(index): {"instance_id": wid, "gpu": self.settings.gpus[index]}
            for index, wid in stack.worker_ids.items()
        }
        self.result["compatibility_digest"] = self.settings.compatibility_digest
        self.result["topology"] = self.settings.topology
        self.save()
        return client

    def cases_phase(self, client: Client) -> None:
        ctx = Context(
            self.stack,
            client,
            self.prompts,
            self.references,
            index_wait=self.args.index_wait,
            telemetry_timeout=self.args.telemetry_timeout,
            recovery_timeout=self.args.recovery_timeout,
            perf_max_tokens=self.args.perf_max_tokens,
            perf_concurrency=self.args.perf_concurrency,
            perf_evict_filler=self.args.perf_evict_filler,
            perf_concurrency_levels=self.args.perf_concurrency_levels,
            hit_block_tokens=self.args.hit_block_tokens,
            baseline_timings=self.baseline_timings,
            engine_cached=self.reference_cached,
            tc02_seed_prefix=self.args.tc02_seed_prefix,
        )
        for name in self.args.cases:
            self.check_deadline()
            result = run_case(name, ctx)
            self.result["cases"].append(result.to_dict())
            self.result["requests"] = client.log
            self.save()
            print(
                json.dumps(
                    {"case": name, "status": result.status, "error": result.error}
                )
            )
            if result.status == FAIL and self.args.stop_on_fail:
                break

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> int:
        blocked_all: str | None = None
        try:
            save_json(self.output / "provenance.json", provenance(self.args))
            self.result["preflight"] = self.stack.preflight()
            self.reference_phase()
            client = self.stack_phase()
            self.cases_phase(client)
        except BlockedError as blocked:
            blocked_all = str(blocked)
        except (Exception, KeyboardInterrupt) as error:
            self.result["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                self.stack.cleanup()
            except Exception as error:  # noqa: BLE001
                self.result["cleanup_error"] = str(error)
        if blocked_all is not None:
            for name in self.args.cases:
                if not any(case["case"] == name for case in self.result["cases"]):
                    self.result["cases"].append(
                        {
                            "case": name,
                            "title": name,
                            "requirement": "",
                            "status": BLOCKED,
                            "checks": [],
                            "evidence": [],
                            "error": blocked_all,
                            "started_at": utc_now(),
                            "finished_at": utc_now(),
                            "measurements": {},
                        }
                    )
        statuses = {case["status"] for case in self.result["cases"]}
        missing = [
            n
            for n in self.args.cases
            if not any(c["case"] == n for c in self.result["cases"])
        ]
        if (
            missing
            or "error" in self.result
            or "cleanup_error" in self.result
            or FAIL in statuses
        ):
            self.result["status"] = FAIL
        elif BLOCKED in statuses:
            self.result["status"] = BLOCKED
        else:
            self.result["status"] = PASS
        self.result["missing_cases"] = missing
        self.result["finished_at"] = utc_now()
        self.save()
        write_junit(self.output, self.result)
        print(
            json.dumps(
                {
                    "status": self.result["status"],
                    "cases": {c["case"]: c["status"] for c in self.result["cases"]},
                    "error": self.result.get("error"),
                    "result": str(self.output / "result.json"),
                }
            )
        )
        if self.result["status"] == PASS:
            return 0
        if self.result["status"] == BLOCKED and self.args.allow_blocked:
            return 0
        return 1


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--mode", choices=MODES, default=GUARD)
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="CUDA device indices: two for --topology agg (0,1), three for disagg (0,1,2; "
        "two-node disagg: 0,0,1 = P0 on GPU0 of node a, P1 on GPU0 of node b, D0 on GPU1 of node a)",
    )
    parser.add_argument("--cases", default="TC-01,TC-02,TC-03,TC-04,TC-05,TC-06")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--reference-http-port", type=int, default=8001)
    parser.add_argument(
        "--pool-gb", type=int, default=2, help="Guard pool size per worker"
    )
    parser.add_argument("--compatibility-digest", default=None)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.8,
        help="vLLM --gpu-memory-utilization per worker (<= 0.45 when both share one GPU)",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--perf-repeats", type=int, default=5, help="perf prompts per arm (TC-05/06)"
    )
    parser.add_argument(
        "--perf-max-tokens", type=int, default=128, help="output tokens per perf request"
    )
    parser.add_argument(
        "--perf-concurrency", type=int, default=1, help="parallel perf requests per arm"
    )
    parser.add_argument(
        "--perf-evict-filler",
        type=int,
        default=0,
        help="TC-05: after the local-hit arm send this many never-seen prompts to "
        "worker 0 to evict its GPU prefix cache, then re-request the perf prompts "
        "so the hit must come from the KVCR DRAM pool (local_deliver); 0 = skip. "
        "Size it above the GPU KV capacity (see --worker-arg=--num-gpu-blocks-override)",
    )
    parser.add_argument(
        "--perf-concurrency-levels",
        default="",
        help="TC-05: comma-separated extra concurrency levels, e.g. 4,8; each level "
        "sends that many fresh prompts in parallel (cold on worker 0, remote hit on "
        "worker 1) and records TTFT/TPOT; tokens are not reference-checked",
    )
    parser.add_argument(
        "--hit-block-tokens",
        type=int,
        default=64,
        help="tokens per prefix-cache hit block: 64 for dense models, 576 for Qwen3.5 "
        "hybrid models (vLLM raises the attention block); hit alignment and the "
        "delivered-pages checks use it",
    )
    parser.add_argument(
        "--tc02-shared-sections",
        type=int,
        default=40,
        help="TC-02 shared prefix P1 length in ~17-token sections (hybrid: 140)",
    )
    parser.add_argument(
        "--tc02-sections",
        type=int,
        default=80,
        help="TC-02 full/partial prompt length in sections (hybrid: 200)",
    )
    parser.add_argument(
        "--tc02-seed-prefix",
        action="store_true",
        help="TC-02: serve the shared prefix P1 alone on worker 0 before the full "
        "prompt (hybrid models: vLLM mamba_cache_mode=align keeps SSM state only at "
        "a request's last aligned boundary, so the P1/P2 junction needs its own request)",
    )
    parser.add_argument(
        "--topology",
        choices=("agg", "disagg"),
        default="agg",
        help="agg: two decode workers (default). disagg: 2P+1D, workers 0/1 are "
        "prefill workers (KVCR source/target) and worker 2 the shared decode worker; "
        "needs three GPUs",
    )
    parser.add_argument(
        "--frontend-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment for the Dynamo frontend process only, e.g. "
        "DYN_LOG=info,dynamo_kv_router=debug,dynamo_llm::kv_router=debug",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32000,
        help="vLLM --max-model-len for every engine (lower it together with "
        "--worker-arg=--num-gpu-blocks-override=N, vLLM requires N*block_size >= this)",
    )
    parser.add_argument(
        "--keep-hybrid-kv-cache-manager",
        action="store_true",
        help="do not pass --disable-hybrid-kv-cache-manager to vLLM (needed for "
        "SSM/linear-attention hybrids such as Qwen3.5)",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="let vLLM use CUDA graphs (much faster decode for GDN/Mamba hybrids)",
    )
    parser.add_argument(
        "--external-services",
        type=Path,
        default=None,
        help="sidecar topology: directory shared with kvcr_service containers holding "
        "memory-<i>.sock and kvcr-service-<i>.log; the harness attaches instead of "
        "launching the services (Guard mode only)",
    )
    parser.add_argument(
        "--remote-worker",
        action="append",
        default=[],
        metavar="INDEX=URL",
        help="two-node topology: run worker INDEX (and expect its kvcr_service sidecar) on "
        "the node whose phase1.node_agent listens at URL; repeatable; requires "
        "--external-services and --etcd",
    )
    parser.add_argument(
        "--etcd",
        default=None,
        metavar="ENDPOINTS",
        help="use Dynamo etcd discovery at these endpoints instead of the file store "
        "(required across nodes: the file store's inotify watch is node-local)",
    )
    parser.add_argument(
        "--worker-arg",
        action="append",
        default=[],
        help="extra dynamo.vllm argument, repeatable (e.g. --worker-arg=--mamba-cache-mode=align)",
    )
    parser.add_argument(
        "--prompt-tag", default=None, help="Fixed tag for reproducible prompts"
    )
    parser.add_argument(
        "--batch-invariant", action="store_true", help="Set VLLM_BATCH_INVARIANT=1"
    )
    parser.add_argument(
        "--no-state-agent",
        action="store_true",
        help="Guard mode: do not run dynamo.kv_state_agent (TC-04 then has no hint)",
    )
    parser.add_argument(
        "--timeout", type=float, default=2400, help="Overall deadline in seconds"
    )
    parser.add_argument("--readiness-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--telemetry-timeout", type=float, default=60)
    parser.add_argument("--recovery-timeout", type=float, default=180)
    parser.add_argument("--index-wait", type=float, default=15)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument(
        "--allow-blocked", action="store_true", help="Exit 0 when only BLOCKED remains"
    )
    args = parser.parse_args(argv)
    gpus = [part.strip() for part in args.gpus.split(",")]
    wanted_gpus = 3 if args.topology == "disagg" else 2
    if len(gpus) != wanted_gpus or not all(part.isdecimal() for part in gpus):
        parser.error(
            f"--gpus needs {wanted_gpus} CUDA device indices for --topology {args.topology}"
        )
    args.gpus = [int(part) for part in gpus]
    if args.topology == "disagg" and args.external_services is None:
        parser.error("--topology disagg needs --external-services (kvcr_service sidecars)")
    # Same index on two nodes (--remote-worker) is two GPUs, not one shared GPU.
    if (args.gpus[0] == args.gpus[1] and not args.remote_worker
            and args.gpu_memory_utilization > 0.45):
        parser.error("two workers on one GPU need --gpu-memory-utilization <= 0.45")
    if not 0.05 <= args.gpu_memory_utilization <= 0.95:
        parser.error("--gpu-memory-utilization must be within 0.05..0.95")
    cases = [part.strip().upper() for part in args.cases.split(",") if part.strip()]
    unknown = [case for case in cases if case not in CASES]
    if unknown or not cases:
        parser.error(f"unknown cases {unknown}; choose from {sorted(CASES)}")
    args.cases = cases
    for name in (
        "timeout",
        "readiness_timeout",
        "request_timeout",
        "telemetry_timeout",
        "recovery_timeout",
    ):
        if not 0 < getattr(args, name) <= 7200:
            parser.error(f"--{name.replace('_', '-')} must be >0 and <=7200 seconds")
    if not 0 <= args.index_wait <= 300:
        parser.error("--index-wait must be between 0 and 300 seconds")
    if not 1 <= args.max_tokens <= 512 or not 1 <= args.pool_gb <= 64:
        parser.error("--max-tokens must be 1..512 and --pool-gb 1..64")
    if not 1 <= args.perf_repeats <= 32 or not 1 <= args.perf_max_tokens <= 2048:
        parser.error("--perf-repeats must be 1..32 and --perf-max-tokens 1..2048")
    try:
        args.perf_concurrency_levels = [
            int(item) for item in args.perf_concurrency_levels.split(",") if item.strip()
        ]
    except ValueError:
        parser.error("--perf-concurrency-levels expects integers like 4,8")
    if any(not 2 <= level <= 64 for level in args.perf_concurrency_levels):
        parser.error("--perf-concurrency-levels entries must be 2..64")
    if args.hit_block_tokens < 1 or args.hit_block_tokens % 64:
        parser.error("--hit-block-tokens must be a positive multiple of 64")
    if not 2 <= args.tc02_shared_sections < args.tc02_sections <= 1000:
        parser.error("--tc02-shared-sections must be < --tc02-sections (<= 1000)")
    if not 0 <= args.perf_evict_filler <= 512:
        parser.error("--perf-evict-filler must be 0..512")
    if not 1 <= args.perf_concurrency <= 64:
        parser.error("--perf-concurrency must be 1..64")
    if args.http_port == args.reference_http_port:
        parser.error("--http-port and --reference-http-port must differ")
    if args.external_services is not None:
        if args.mode != GUARD:
            parser.error("--external-services requires --mode guard")
        if not args.compatibility_digest:
            parser.error("--external-services needs --compatibility-digest matching the sidecars")
    remote: dict[int, str] = {}
    for item in args.remote_worker:
        index, sep, url = item.partition("=")
        if not sep or index not in ("0", "1") or not url.startswith("http"):
            parser.error(f"--remote-worker expects 0=http://host:port or 1=http://host:port, got {item!r}")
        remote[int(index)] = url
    args.remote_workers = remote
    if remote:
        if args.external_services is None:
            parser.error("--remote-worker needs --external-services (the remote kvcr_service is a sidecar)")
        if not args.etcd:
            parser.error("--remote-worker needs --etcd (file discovery does not cross nodes)")
        if len(remote) == 2:
            parser.error("at least one worker must run on the harness node")
    return args


def main() -> int:
    args = arguments()
    try:
        runner = Runner(args)
    except Exception as error:  # noqa: BLE001
        print(f"Cannot initialize run: {error}", file=sys.stderr)
        return 1

    def interrupted(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = {
        n: signal.signal(n, interrupted) for n in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return runner.run()
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
