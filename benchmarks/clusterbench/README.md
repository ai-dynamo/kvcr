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
  control channel, the bytes moved, one OS process per engine rank, and
  vLLM's own scheduler and KV-offload tiering, which make every KVCR call.
- **Stand-ins:** the model and GPU memory (the forward pass is a sleep that
  releases the GIL, and nothing is copied between device and CPU) and Dynamo
  (a small router with Dynamo's hint rule).

## Running

The benchmark needs the vLLM commit pinned in `requirements.txt`, built
without its CUDA extensions. From a KVCR checkout, in an environment of its
own:

```bash
uv venv .venv-clusterbench && source .venv-clusterbench/bin/activate
VLLM_TARGET_DEVICE=empty CARGO=false uv pip install -e . --group dev \
  -r benchmarks/clusterbench/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
export PYTHONPATH=benchmarks

# Four workers and a coordinator on this machine. Nodes are only labels here.
python -m clusterbench local \
  --config benchmarks/clusterbench/scenarios/smoke.toml --nodes 2 --per-node 2

# Compare two KVCR checkouts (or one with itself, for the noise floor).
python -m clusterbench ab \
  --config benchmarks/clusterbench/scenarios/agentic-stress.toml \
  --nodes 2 --per-node 2 --cores-per-worker 4 --reps 3 \
  --arm base=/path/to/kvcr-main --arm cand=/path/to/kvcr-branch
```

An arm is a KVCR checkout (the directory holding `src/kvcr`) that the pinned
vLLM's KVCR tier can import; a run fails if its workers loaded KVCR from
anywhere else. The harness and vLLM are the same for every arm, so every arm
is driven identically. Local numbers show correctness and rough direction;
quote cluster numbers.

Any option can be set with `--set section.field=value`; `config.py` documents
them all, and an option it does not have is refused. Tests:
`python -m pytest tests/clusterbench` (without vLLM, those that need it skip).

## Scenarios

| File | What it is |
|---|---|
| `smoke.toml` | Seconds long, tiny tiers, every delivered byte checked |
| `agentic.toml` | The baseline: 8 sessions per rank, one turn in twenty migrates |
| `agentic-stress.toml` | Twice the sessions, fast turns, a quarter migrating, a KVCR pool that evicts |
| `agentx-fast.toml` | The AgentX trace with the modelled GPU, answers and waits compressed; needs `workload.trace` |

With realistic GPU times (`engine.step_ms`, `engine.prefill_us_per_token`) a
rank finishes one or two requests a second; `agentx-fast` shrinks them until
KVCR's time on the engine thread is what limits a rank. Blocks are 16 KiB
(`tiers.block_bytes`), far smaller than a real KV block, so that many ranks
fit on a node: host cost follows the number of blocks, not their size. A
block is a whole number of pages: a host with 64 KiB pages needs
`tiers.block_bytes=65536`.

## How it works

- **Coordinator:** generates the workload (closed-loop coding-agent sessions,
  synthetic or replayed from a trace), routes it and drives the phases
  (register, configure, mesh warm, fill, measure, stop). It never imports
  `kvcr`, so it is the same program for every arm. In mesh warm every worker
  pulls one block from every other, so the window holds no first-contact
  connection setup.
- **Workers:** one process each, hosting vLLM's `Scheduler` with its
  offloading connector and KVCR tier, which build the `KVCR` and decide every
  call to it. KVCR's telemetry is on (vLLM's default is off): the report
  reads it.
- **Driver** (`engine_vllm.py`): fakes only what needs a GPU. It gives vLLM
  requests with block ids for hashes, sleeps for the forward pass, and reports
  the connector's copy jobs as finished. `kvcr_probe.py` sits between vLLM and
  the `KVCR` object: it times the calls, counts what moved and checks bytes.
- **Router:** a session stays on its worker and moves with
  `router.migration_prob` per turn. After choosing a worker the router hints
  the peer holding the longest prefix, if that beats what the chosen worker
  holds. That is Dynamo's rule.

## Trace replay

`workload.trace` replays recorded sessions: the AgentX trace
(`semianalysisai/cc-traces-weka-062126` on Hugging Face, 393 Claude Code
sessions, 68,266 requests). Compile it once, then pass the file to the run:

```bash
python -m clusterbench trace traces.jsonl -o agentx-062126.cbtrace
python -m clusterbench local \
  --config benchmarks/clusterbench/scenarios/agentx-fast.toml \
  --set workload.trace=$PWD/agentx-062126.cbtrace --nodes 2 --per-node 4
```

The replay follows aiperf's AgentX timing, sub-agents included; the `trace.py`
docstring has the rules. `workload.sessions_per_worker` sessions play at once
per worker, each with block ids of its own. A run is refused if its longest
request does not fit `tiers.gpu_blocks` or its longest prompt
`tiers.primary_blocks`.

## Results

| In the run directory | Content |
|---|---|
| `summary.json`, `reports/` | What `report` prints; one raw report per worker |
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
microseconds per moved block; KVCR time per engine step. A moved block is one
KVCR copied: a delivered block, or a deposited block its pool did not already
hold. Throughput and latency are reported too, but they depend on the
closed-loop workload and the modelled engine as much as on KVCR. The engine
thread's time is split into KVCR calls, vLLM's own scheduling, the modelled
forward pass, idle and the rest ("harness").

**Failed blocks.** Hinted blocks that a peer did not deliver are counted by
the status KVCR returns for them: `timeout`, `unavailable` (the source no
longer held the block, or could not pin it), `aborted` or `failed`.
`report --detail` has KVCR's reasons. Blocks *beyond the hint* are counted
apart: vLLM's opportunistic query asks the peer for the rest of the prompt
too, which it cannot have.

**INVALID.** A run is marked INVALID, with the reason, when:

- delivered bytes did not match, or no request completed;
- the coordinator dispatched late inside the window (p99 over 1 ms) or had no
  slack;
- on any worker, the engine or progress thread waited for a CPU more than 1%
  of the window;
- peers first connected inside the window after a full mesh warm, or
  mesh-warm pulls failed;
- a host tier had not filled to `phases.min_primary_occupancy` when the window
  opened;
- KVCR raised resilience events, or operations were still in flight at exit;
- the run was profiled: `CB_PROFILE_DIR=<dir>` makes every worker write a
  cProfile of the window, which slows it.

`report --arm` compares valid runs only, fails if an arm has none, and warns
when arms were offered different amounts of work.

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

- `$CB_ROOT/site` holds the Python packages. Fill an empty one on a compute
  node (Python 3.11 or later, git; `pip.pyz` from bootstrap.pypa.io works where
  there is no pip): `VLLM_TARGET_DEVICE=empty CARGO=false python3 -m pip install
  --target $CB_ROOT/site --extra-index-url https://download.pytorch.org/whl/cpu
  -r $S/requirements.txt nixl`.
- `CB_NET_DEVICES` is one RDMA device per socket and `CB_CTRL_IF` the
  interface of KVCR's control channel. Workers pin themselves by socket.
- `CB_HARNESS` selects the harness copy a cell runs (default
  `$CB_ROOT/harness`); give a queued job its own, so a restage cannot change it.

## Not covered

- Kubernetes launch, the G3 disk tier, the Guard service, and `kv.prefetch`
  and `kv.move` hints.
- Engines other than vLLM, such as SGLang's direct linker.
- Forcing evictions or timeouts: they are reported, not injected.
- A pin other than the one in `requirements.txt`: the driver uses vLLM's
  scheduler at the level its own tests do, so a newer vLLM may need changes.
- Load-aware routing such as Dynamo's cost formula: sessions are sticky and
  migrate at a fixed rate.
- In trace replay, a sub-agent's requests run one after another (aiperf splits
  some into parallel streams), a sub-agent recorded as starting during its
  parent turn starts when that turn returns, and the answer blocks that never
  reappear in a prompt are not cached.
