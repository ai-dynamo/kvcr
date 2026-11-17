<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# clusterbench: KVCR at deployment scale, without GPUs

`clusterbench` runs many real KVCR instances that pull KV blocks from each
other, the way engine ranks do in a large deployment, with nothing but CPUs.
It exists to measure KVCR's host-side cost (the engine thread's time inside
KVCR calls, the progress thread, peer control, transfer latency under many
peers) and to compare KVCR source trees against each other.

What is real and what is not:

| Real | Stand-in |
|---|---|
| `KVCR`, its progress thread, the NIXL agent and UCX transfers | The engine: a fake that calls KVCR in vLLM's pattern, or in the pattern of SGLang's direct linker |
| The ZMQ peer control channel between instances | GPU memory: bookkeeping only (vLLM profile) or host memory (SGLang profile) |
| The bytes moved (host memory, any block size) | Dynamo: a small router with Dynamo's hint rule |
| One OS process per engine rank, with its own GIL | The model: a sleep that releases the GIL |

## Quick start

Everything runs from a KVCR checkout with its environment set up (`uv sync`).

```bash
PYTHONPATH=benchmarks uv run python -m clusterbench local \
  --config benchmarks/clusterbench/scenarios/smoke.toml --nodes 2 --per-node 2
```

That starts four workers and a coordinator on this machine, connects every
pair, runs sessions for a few seconds and prints a summary. "Nodes" are only
labels locally; every transfer goes over UCX shared memory or TCP.

Compare two KVCR trees (or the same tree twice, for the noise floor):

```bash
PYTHONPATH=benchmarks uv run python -m clusterbench ab \
  --config benchmarks/clusterbench/scenarios/agentic-stress.toml \
  --nodes 2 --per-node 2 --cores-per-worker 4 --reps 3 \
  --arm base=/path/to/kvcr-main --arm cand=/path/to/kvcr-branch
```

Each arm is a KVCR checkout (the directory holding `src/kvcr`). The harness
itself always comes from this checkout, so both arms are driven identically.
Local numbers are only as quiet as the machine; use them for correctness and
rough direction, and the cluster for anything you intend to quote.

## How a run works

- **Workers** each embed a real `KVCR`. An anonymous mapping stands in for
  vLLM's primary CPU offload tier (KVCR's framework region) and, optionally, a
  second one is the KVCR-owned DRAM pool. A fake engine drives KVCR from the
  main thread.
- **The coordinator** generates the workload and routes it. It never imports
  `kvcr`, so it is the same program for every arm.
- **Phases**: register, configure, mesh warm, fill, measure, stop. During mesh
  warm every worker pulls one block from every other worker, so the measured
  window does not consist of first-contact connection setup. With 144 workers
  there are about 20,000 ordered pairs.

### The fake engine

`engine_vllm.py` mirrors vLLM's KVCR tier
(`vllm/v1/kv_offload/tiering/kvcr/manager.py` and the tiering manager and
offloading scheduler around it). What KVCR sees is what it sees from vLLM:

- `query` one key at a time, only when the CPU tier misses, walking to the end
  of the prompt;
- one `deliver` per request per step, into CPU-tier slots; never `fetch`;
- a `deposit` for every block stored from the device to the CPU tier;
- `align_sequence` over the request's whole prefix after every completed
  deliver or deposit and at finish, then `discard_hint`;
- `poll_completed` at least once per step, `get_stats` every step, an idle
  engine still stepping about once a millisecond;
- framework pins resolved once per step from the CPU tier.

`tests/clusterbench/test_cb_engine_vllm.py` pins this call sequence. If vLLM's
tier changes, change the engine and those tests together.

### The SGLang direct-linker profile

`engine.profile = "sglang"` (add `--set engine.profile=sglang` to any
scenario) swaps in SGLang's KVCR direct linker, from SGLang PR #45650 at the
revision KVCR's quick start pins. `linker_port.py` is the linker's owner
thread, ported without torch; `engine_sglang.py` is the scheduler around it;
`device_tree.py` is the device radix cache. KVCR sees something quite
different from the vLLM tier:

- A dedicated `kvcr-owner` thread makes every KVCR call. The scheduler thread
  blocks on each linker call. While a load or a backup is pending the owner
  spins, calling `poll_completed` on every iteration, and the scheduler does
  not sleep when idle, so an idle rank still polls.
- A page (`tokens_per_block` tokens) is one key with a K and a V piece per
  layer: `2 x tiers.layers` pieces, each its own labelled region and its own
  slot in one KVCR pool. `tiers.block_bytes` is the bytes of a whole page, so
  a page moves the same bytes as a vLLM-profile block.
- Lookup is one batched `query` over every page the device lacks.
- A restore is one `deliver` per layer per request, straight into device
  memory, at most four outstanding. The scheduler waits for the whole batch.
  If any key fails, every request loading in that batch is aborted; nothing
  is recomputed.
- A backup is `deposit` calls of at most 4096 pieces, one at a time. Every
  new tree node is backed up, including pages just restored from KVCR.
- No `align_sequence`, no source pins: peers read KVCR-owned memory only.

Here the device cache is real memory (`tiers.gpu_blocks` pages), because KVCR
reads and writes it, and there is no CPU primary tier. Device regions are
host memory, so this measures KVCR's host path, not GPU transfers. Only the
dense, TP1 layout is modelled.

`tests/clusterbench/test_cb_engine_sglang.py` pins the call pattern. If the
linker changes, change the port and those tests together.

### What-if switches

A few `engine.*` options make a fake engine behave differently from the real
integration, to estimate what a change on the engine's side would be worth
before anyone writes it. Their defaults are what the engines do today, and a
run with any of them set does not describe a real deployment:

| Option | Default | What it changes |
|---|---|---|
| `align_on_finish_only` | `false` | vLLM: `align_sequence` only when a request finishes |
| `linker_layers_per_deliver` | `1` | SGLang: layers named by one `deliver` |
| `linker_max_deliveries` | `4` | SGLang: `deliver` operations outstanding at once |
| `linker_backup_restored` | `true` | SGLang: `false` skips the backup of pages just restored from KVCR |
| `linker_deposit_pieces` | `4096` | SGLang: pieces per `deposit` (32 pages of a 64-layer model) |

With vLLM's opportunistic query, every block of a hinted request that is not
held locally is requested from the peer, including the new part of the prompt
the peer cannot have. Those requests fail by design; the report counts them
separately from failures of blocks the router said the peer holds.

### Workload and router

Sessions are coding-agent conversations: a system and tool prefix shared by all
sessions of an agent type, then a context that grows every turn until it is
compacted at the cap. Sizes and think times are lognormal, with defaults taken
from the Claude Code traces the team replays. Each session draws from its own
random stream, so what a session asks for does not depend on how fast the
system under test is.

The router keeps, per worker and tier, how much of each stream's root-aligned
prefix it holds, fed by the workers. After choosing a worker it attaches a
`kv.fetch` hint naming the peer with the longest prefix, if that beats what
the chosen worker holds, exactly as Dynamo does. Two ways to choose:

- `sticky` (default): a session stays on its worker and moves with
  `router.migration_prob` per turn, to the worker with the fewest sessions.
  Whether a session moves does not depend on how fast the system is, so both
  arms of an A/B are offered the same work.
- `cost`: Dynamo's default KV-router formula.

### Trace replay

Instead of generating sessions, the coordinator can replay recorded ones: the
AgentX trace, which is SemiAnalysis's corpus of Claude Code sessions in the
"Weka" format (`semianalysisai/cc-traces-weka-062126` on Hugging Face: 393
sessions, 68,266 requests, 58% of them made by sub-agents). Each request
carries the hashes of the 64-token blocks of its whole prompt, so the replay
has the real contexts (1,400 blocks at the median, 256k tokens at most), the
real branches and compactions, and the real answer lengths and waits.

Compile the corpus once (a few seconds), then give the file to any scenario:

```bash
python -m clusterbench trace traces.jsonl -o agentx-062126.cbtrace
python -m clusterbench local --config benchmarks/clusterbench/scenarios/agentx-fast.toml \
  --set workload.trace=$PWD/agentx-062126.cbtrace --nodes 2 --per-node 4
```

The compiled file is a few megabytes and is all a run needs; only the
coordinator reads it. `trace` prints what to size a scenario by (blocks per
request, new blocks per turn, how many sub-agents start at once).

How a session is replayed, following aiperf's AgentX rules:

- Closed loop. A request goes out the recorded gap after the request it
  depends on has finished (`workload.trace_think_scale` multiplies the gap,
  `workload.think_cap_s` caps it), so the offered work does not depend on how
  fast the system is.
- Sub-agents start after the parent turn before them and run side by side;
  the parent's next turn waits for all of them. The router places a sub-agent
  as a new session, on the least busy worker: it shares no block with its
  parent.
- `workload.sessions_per_worker` is the number of sessions replayed at once
  per worker. When one ends, the next one starts; a session replayed again
  gets block ids of its own, so nothing is shared between plays (AgentX's
  cache-bust rule). With `workload.aged_start` the first sessions start
  part-way through their recording.
- A request's blocks are the runs of the session's prefix tree on its path.
  The blocks its answer adds are the first blocks of the next request that
  extends its prompt.

What the replay does not reproduce:

- The requests inside one sub-agent run one after another; aiperf splits some
  sub-agents further into parallel streams.
- About 13% of answer blocks never appear in a later prompt (thinking, for example).
  They are not cached, which slightly understates device-cache pressure.
- Where a later request branches off is known in advance, so on the SGLang
  profile a cold prompt enters the tree as one node per run where SGLang would
  insert one node and split it later.
- No block is shared between sessions. That is true of the trace itself: its
  hashes are local to a session.

The trace must fit the tiers: the coordinator refuses a run whose longest
request exceeds `tiers.gpu_blocks`, or on the vLLM profile
`tiers.primary_blocks`, because the modelled engine would never start it.

## Scenarios

| File | What it is |
|---|---|
| `smoke.toml` | Seconds long, tiny tiers, every delivered byte checked |
| `agentic.toml` | The baseline: 8 sessions per rank, one turn in twenty migrates |
| `agentic-stress.toml` | Twice the sessions, fast turns, a quarter migrating, a KVCR pool that evicts |
| `agentic-stress-fast.toml` | `agentic-stress` with the modelled GPU and think time compressed, so a rank's engine thread is the limit rather than its GPU |
| `peer-only.toml` | No KVCR pool: peers are served from pinned engine memory |
| `agentx.toml` | The AgentX trace with the modelled GPU and recorded waits (capped at 10 s); needs `workload.trace` |
| `agentx-fast.toml` | The AgentX trace with the GPU, the answers and the waits compressed, twice the sessions and half the KVCR pool, so KVCR limits a rank; needs `workload.trace` |

How many requests a rank completes is set by the modelled GPU
(`engine.step_ms`, `engine.prefill_us_per_token`) and the think time, and the
peer-fetch rate is that times the share of turns that migrate
(`router.migration_prob`). With realistic GPU times a rank finishes one or two
requests a second; `agentic-stress-fast` shrinks them until KVCR's time on the
engine thread is what bounds the rank.

Any field can be overridden with `--set section.field=value`; `config.py`
documents them all. Two worth knowing:

- `tiers.block_bytes` is the bytes KVCR moves per block. The default (16 KiB)
  is far smaller than a real KV block so that many ranks fit on a node; host
  cost scales with the number of blocks, not their size. Raise it for a
  data-plane calibration.
- `tiers.tokens_per_block` (64) sets how many blocks a context is.

## Reading the result

A run directory holds `summary.json` (what `report` prints), `reports/` (one
raw report per worker), `timeseries.jsonl`, `workers.json` and the config.

```bash
PYTHONPATH=benchmarks python3 -m clusterbench report runs/my-run
PYTHONPATH=benchmarks python3 -m clusterbench report --detail runs/my-run   # + KVCR telemetry
PYTHONPATH=benchmarks python3 -m clusterbench report \
  --arm base=runs/base-0,runs/base-1 --arm cand=runs/cand-0,runs/cand-1
```

The numbers to compare between arms are the normalised ones:

- engine-thread microseconds per item for each KVCR call;
- engine, progress-thread and native-thread microseconds per moved block;
- KVCR time per engine step (p50, p99).

The summary also splits each rank's engine thread four ways: time inside KVCR
calls, the fake engine's own bookkeeping ("harness"), the modelled forward
pass, and idle steps. KVCR is the bottleneck when its share dominates; if the
harness share is not small next to it, the harness has regressed.

In the SGLang profile the KVCR calls are on the owner thread, so there are two
splits. The owner thread: time inside KVCR calls, the linker's own work,
yielding while it spins, and idle. The scheduler thread: blocked on linker
calls, waiting for loads to land, harness, the modelled forward pass, and
idle. The per-call table and the normalised rows mean the same in both
profiles; empty polls are listed apart as `poll_empty`, since their number
follows wall time rather than work. Aborted requests are counted apart and
are not in requests per second.

Set
`CB_PROFILE_DIR` to have every worker write a cProfile of the measured window
there (it covers the progress thread too, and slows the run).

Throughput and latency are reported too, but they depend on the closed-loop
workload and the modelled engine as much as on KVCR. In particular
`cache_resolve` (first lookup to request start) includes waiting for CPU-tier
slots when the tier is smaller than the contexts being pulled into it;
`deliver_remote` is the latency of the KVCR operation itself.

`report` needs only the standard library, so it runs on a login node whose
architecture differs from the compute nodes.

A run is marked INVALID, with the reason, when its numbers should not be
trusted:

- delivered bytes did not match, or a worker did not report;
- the coordinator dispatched late (p99 over 1 ms) or had no slack, so it
  rather than the workers may have set the pace;
- an engine, owner or progress thread waited for a CPU more than 1% of the
  window;
- peers connected for the first time inside the window after a full mesh warm;
- a host tier (vLLM's CPU tier, or KVCR's pool in the SGLang profile) had not
  filled to `phases.min_primary_occupancy` when the window opened (the fill
  phase was too short);
- KVCR raised resilience events, or operations were still in flight at exit.

`report --arm` also warns when the arms were offered different amounts of work.

## Slurm

The scripts in `slurm/` hold nodes with a keeper job and run cells as job
steps, so an A/B can alternate arms inside one allocation.

```bash
export CB_ROOT=/shared/fs/kvcr-clusterbench      # on the cluster
# From your workstation: copy the harness and the arms.
CB_HOST=login-node benchmarks/clusterbench/slurm/stage.sh $CB_ROOT \
  base=main cand=my-branch
# On the login node: hold nodes, look at one, run cells.
sbatch --nodes=4 --time=01:00:00 --account=ACCOUNT --partition=PARTITION \
  --job-name=NAME --output=$CB_ROOT/logs/keeper-%j.out \
  --export=ALL,CB_ROOT=$CB_ROOT $CB_ROOT/harness/benchmarks/clusterbench/slurm/keeper.sbatch
S=$CB_ROOT/harness/benchmarks/clusterbench
srun --jobid $JOB --overlap -N1 -n1 bash $S/slurm/nodeinfo.sh
export CB_CTRL_IF=ib0 CB_NET_DEVICES=mlx5_0:1,mlx5_1:1
bash $S/slurm/run_cell.sh $JOB base first-cell 4 8 $S/scenarios/agentic.toml
bash $S/slurm/ab.sh $JOB myab 4 8 $S/scenarios/agentic-stress.toml 3 "base cand"
touch $CB_ROOT/control/release-$JOB
```

`$CB_ROOT/site` must hold `nixl`, `msgspec` and `pyzmq` built for the compute
nodes (`pip install --target $CB_ROOT/site ...` on one of them).

`CB_HARNESS` selects the harness copy a cell runs (default `$CB_ROOT/harness`).
Stage a copy under another name and export it in a job that will sit in the
queue, so restaging the default later does not change what that job runs.

Each worker pins itself: workers alternate between sockets, each takes a block
of physical cores, its engine thread and KVCR's progress thread (and the
linker's owner thread in the SGLang profile) get a core each, and NIXL's
threads share the rest. Set `CB_NET_DEVICES` to one RDMA
device per socket and `CB_CTRL_IF` to the interface KVCR's control channel
should advertise.

## Not covered yet

Kubernetes launch, SGLang's HiCache storage backend and its TP>1 or
sliding-window layouts, `kv.prefetch` and
`kv.move` hints (Dynamo does not emit them today), open-loop replay by
recorded timestamp, a recorded initial placement, the G3 disk tier, the Guard
service and fault injection.
