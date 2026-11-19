<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# clusterbench: KVCR at deployment scale, without GPUs

Runs many real KVCR instances that pull KV blocks from each other, on CPUs
only. Use it to measure KVCR's host-side cost (engine-thread time inside KVCR
calls, the progress thread, peer control, transfer latency with many peers)
and to compare KVCR source trees.

- **Real:** `KVCR`, its progress thread, NIXL/UCX transfers, the ZMQ peer
  control channel, the bytes moved, one OS process per engine rank.
- **Stand-ins:** the engine (a fake that makes vLLM's KVCR calls, or those of
  SGLang's direct linker), GPU memory, the model (a sleep that releases the
  GIL) and Dynamo (a small router with Dynamo's hint rule).

## Running

From a KVCR checkout with its environment set up (`uv sync`):

```bash
# Four workers and a coordinator on this machine. Nodes are only labels here.
PYTHONPATH=benchmarks uv run python -m clusterbench local \
  --config benchmarks/clusterbench/scenarios/smoke.toml --nodes 2 --per-node 2

# Compare two KVCR checkouts (or one with itself, for the noise floor).
PYTHONPATH=benchmarks uv run python -m clusterbench ab \
  --config benchmarks/clusterbench/scenarios/agentic-stress.toml \
  --nodes 2 --per-node 2 --cores-per-worker 4 --reps 3 \
  --arm base=/path/to/kvcr-main --arm cand=/path/to/kvcr-branch
```

An arm is a KVCR checkout (the directory holding `src/kvcr`); a run fails if
its workers loaded KVCR from anywhere else. The harness always comes from this
checkout, so every arm is driven identically. Local numbers show correctness
and rough direction; quote cluster numbers.

Any option can be set with `--set section.field=value`; `config.py` documents
them all, and an option it does not have is refused. Tests:
`uv run pytest tests/clusterbench`.

## Scenarios

| File | What it is |
|---|---|
| `smoke.toml` | Seconds long, tiny tiers, every delivered byte checked |
| `agentic.toml` | The baseline: 8 sessions per rank, one turn in twenty migrates |
| `agentic-stress.toml` | Twice the sessions, fast turns, a quarter migrating, a KVCR pool that evicts |
| `agentic-stress-fast.toml` | `agentic-stress` with the modelled GPU and think time compressed |
| `peer-only.toml` | No KVCR pool: peers are served from pinned engine memory |
| `agentx.toml` | The AgentX trace with the modelled GPU and recorded waits; needs `workload.trace` |
| `agentx-fast.toml` | The AgentX trace compressed, with twice the sessions and half the KVCR pool; needs `workload.trace` |

With realistic GPU times (`engine.step_ms`, `engine.prefill_us_per_token`) a
rank finishes one or two requests a second. The `-fast` scenarios shrink them
until KVCR's time on the engine thread is what limits a rank.

| Option | Default | Meaning |
|---|---|---|
| `engine.profile` | `vllm` | `sglang` swaps in SGLang's direct linker |
| `router.migration_prob` | 0.05 | Chance per turn that a session moves, to the worker with the fewest sessions, whatever the system's speed |
| `tiers.block_bytes` | 16 KiB | Bytes KVCR moves per block. Far smaller than a real KV block so that many ranks fit on a node; host cost follows the number of blocks, not their size |
| `tiers.tokens_per_block` | 64 | Tokens per block |
| `workload.trace` | none | Replay a compiled trace instead of generating sessions |

## How it works

- **Coordinator:** generates the workload, routes it and drives the phases
  (register, configure, mesh warm, fill, measure, stop). It never imports
  `kvcr`, so it is the same program for every arm. In mesh warm every worker
  pulls one block from every other, so the window holds no first-contact
  connection setup.
- **Workers:** one process each, with a real `KVCR` and a fake engine driving
  it. Host mappings stand in for engine memory and KVCR's DRAM pool.
- **vLLM profile** (`engine_vllm.py`): vLLM's KVCR tier. One `query` per key
  on a CPU-tier miss, one `deliver` per request per step, a `deposit` per
  stored block, `align_sequence` after each. A request runs once the device
  has room for all of it.
- **SGLang profile** (`engine_sglang.py`, `linker_port.py`, `device_tree.py`):
  the direct linker of SGLang PR #45650. An owner thread makes every KVCR
  call: a batched `query`, one `deliver` per layer (a layer computes while
  the later ones load), `deposit` batches, no `align_sequence`. A failed key
  aborts the requests loading in its batch.
- **Router:** tracks how much of each stream's prefix every worker holds.
  After choosing a worker it hints the peer with the longest prefix, if that
  beats what the chosen worker holds. That is Dynamo's rule.
- **Workload:** synthetic coding-agent sessions (a shared system prefix, a
  context that grows each turn and is compacted at the cap) or a replayed
  trace. Both are closed loop.

The exact call sequence of each profile is in the module docstrings and pinned
by `tests/clusterbench/test_cb_engine_*.py`. If the upstream integration
changes, change the engine and its tests together.

## Trace replay

`workload.trace` replays recorded sessions: the AgentX trace
(`semianalysisai/cc-traces-weka-062126` on Hugging Face, 393 Claude Code
sessions, 68,266 requests). Compile it once, then pass the file to an `agentx`
scenario:

```bash
python -m clusterbench trace traces.jsonl -o agentx-062126.cbtrace
python -m clusterbench local --config benchmarks/clusterbench/scenarios/agentx-fast.toml \
  --set workload.trace=$PWD/agentx-062126.cbtrace --nodes 2 --per-node 4
```

The replay follows aiperf's AgentX rules:

- A request goes out the recorded gap after the one it depends on finishes
  (`workload.trace_think_scale` scales the gap, `workload.think_cap_s` caps
  it).
- Sub-agents run side by side as sessions of their own. The first parent turn
  recorded at or after a sub-agent's end waits for it; earlier ones do not.
- `workload.sessions_per_worker` sessions play at once per worker. Each play
  gets block ids of its own, so plays share nothing.

A run is refused if its longest request does not fit `tiers.gpu_blocks` (or
`tiers.primary_blocks` on the vLLM profile).

## Results

| In the run directory | Content |
|---|---|
| `summary.json` | What `report` prints |
| `reports/` | One raw report per worker |
| `requests.jsonl` | One row per request: where it went, the hint, the outcome |
| `timeseries.jsonl`, `workers.json`, `config.json` | Periodic worker stats, who ran where, the resolved config |

```bash
PYTHONPATH=benchmarks python3 -m clusterbench report runs/my-run
PYTHONPATH=benchmarks python3 -m clusterbench report --detail runs/my-run   # + KVCR telemetry
PYTHONPATH=benchmarks python3 -m clusterbench report \
  --arm base=runs/base-0,runs/base-1 --arm cand=runs/cand-0,runs/cand-1
```

`report` needs only the standard library, so it runs on a login node of
another architecture.

**What to compare between arms.** The normalised numbers: engine-thread
microseconds per item for each KVCR call; engine, progress and native-thread
microseconds per moved block; KVCR time per engine step. Throughput and
latency are reported too, but they depend on the closed-loop workload and the
modelled engine as much as on KVCR.

**Thread split.** Each rank's engine thread is split into KVCR calls, the fake
engine's own bookkeeping ("harness"), the modelled forward pass and idle. KVCR
is the bottleneck when its share dominates. On the SGLang profile the KVCR
calls are on the owner thread, which gets a split of its own.

**Failed blocks.** Hinted blocks that a peer did not deliver are counted by
the status KVCR returns for them: `timeout`, `unavailable` (the source no
longer held the block, or could not pin it), `aborted` or `failed`.
`report --detail` has KVCR's reasons. A KVCR tree that returns no reason shows
them as `unclassified`. Blocks *beyond the hint* are counted apart: vLLM's
opportunistic query asks the peer for the rest of the prompt too, which it
cannot have.

**Request log.** `requests.jsonl` starts with a header naming its fields:
session, turn, worker, hint source (-1: none), hint blocks, send time, whether
the window was open, prompt blocks, blocks the router believed the worker
held, then the outcome (blocks delivered by a peer, failed and timed out, load
and first-token times, aborted). A request still in flight at the end has no
outcome.

**INVALID.** A run is marked INVALID, with the reason, when:

- delivered bytes did not match, or a worker did not report;
- the coordinator dispatched late (p99 over 1 ms) or had no slack;
- an engine, owner or progress thread waited for a CPU more than 1% of the
  window;
- peers first connected inside the window after a full mesh warm;
- a host tier had not filled to `phases.min_primary_occupancy` when the window
  opened;
- KVCR raised resilience events, or operations were still in flight at exit.

`report --arm` compares valid runs only and fails if an arm has none. It warns
when arms were offered different amounts of work.
`CB_PROFILE_DIR=<dir>` makes every worker write a cProfile of the window (it
slows the run).

## Slurm

`slurm/` holds nodes with a keeper job and runs cells as job steps, so an A/B
alternates arms inside one allocation.

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

- `$CB_ROOT/site` must hold `nixl`, `msgspec` and `pyzmq` built for the
  compute nodes (`pip install --target $CB_ROOT/site ...` on one of them).
- `CB_NET_DEVICES` is one RDMA device per socket and `CB_CTRL_IF` the
  interface KVCR's control channel advertises. Workers pin themselves,
  alternating between sockets.
- `CB_HARNESS` selects the harness copy a cell runs (default
  `$CB_ROOT/harness`). Give a queued job a copy of its own, so that a later
  restage does not change what it runs.

## Not covered

- Kubernetes launch, the G3 disk tier, the Guard service, and `kv.prefetch`
  and `kv.move` hints.
- SGLang's HiCache storage backend and its TP>1 and sliding-window layouts.
- Forcing evictions or timeouts: they are reported, not injected.
- vLLM's preemption: it allocates device blocks as a request grows and
  preempts when they run out; here a request waits until all of it fits.
- Load-aware routing such as Dynamo's cost formula: sessions are sticky and
  migrate at a fixed rate.
- In trace replay, a sub-agent's requests run one after another (aiperf splits
  some into parallel streams), and the 13% of answer blocks that never
  reappear in a prompt are not cached.
- Replaying a recorded placement. It was tried and dropped: plain repeats
  already agree within 1% on a cluster, and a fixed placement is unstable at
  saturation, because nothing moves load off a worker that falls behind
  (throughput fell 45% on 32 workers).
