# KVCR Developer Guide

This guide is for developers who are comfortable with Python, Linux, and
inference engines such as vLLM, but are new to KV Cache Runner (KVCR) and
Dynamo. It covers the local development path from environment setup through
standalone validation.

To evaluate KVCR with an existing Dynamo and vLLM stack without changing its
source, follow the [quick start](quick-start.md) instead.

---

## Prerequisites

Use a Linux development environment with:

- Python 3.10 or newer;
- [`uv`](https://docs.astral.sh/uv/);
- a C/C++ runtime compatible with the NIXL wheel selected by the project; and
- enough local memory and disk space for the tests you intend to run.

KVCR declares its Python dependencies in `pyproject.toml`. In particular, it
pins a compatible NIXL version. Let `uv` resolve that dependency instead of
installing a different NIXL release manually.

The framework-neutral unit suite does not require a GPU. Tests for a concrete
framework adapter, CUDA-aware NIXL transport, or cross-worker KV transfer may
require GPUs and the native dependencies of that framework.

---

## Understand KVCR

### Component boundaries

KVCR is an in-process cache runtime, not a request router and not a standalone
inference server. A typical integration contains the following components:

| Component | Responsibility |
| --- | --- |
| Framework or inference engine | Owns GPU memory, request scheduling, KV block allocation, and framework-memory lifetime |
| KVCR | Manages KVCR-owned tiers, cache policy, request-scoped source hints, and asynchronous local or remote data movement |
| Dynamo KV router | Maintains an eventually consistent system-wide KV inventory, selects a worker, and supplies source hints when remote reuse is useful |
| NIXL | Executes the payload transfer between memory or storage descriptors |
| Peer control channel | Exchanges connection metadata, destination descriptors, acknowledgements, and transfer-control messages between KVCR instances |
| KVCR service | Owns shared KVCR DRAM pools independently of a worker process |

The router is on the **control path**, not the data path. It tells a destination
worker where useful KV may exist. The source and destination KVCR instances
coordinate the operation, while NIXL moves the payload directly. Payload bytes
do not pass through Dynamo.

The framework remains the sole owner of its GPU memory. KVCR may transfer to or
from framework-provided descriptors, but it does not independently allocate,
evict, or free framework GPU blocks. Framework-owned source memory must remain
pinned until KVCR releases it.

### KV ownership and tiers

KVCR distinguishes framework-owned memory from KVCR-owned storage:

- **Framework memory** is allocated and scheduled by the engine. KVCR accesses
  it only through descriptors and the framework pinning interface.
- **KVCR local DRAM** is a bounded KVCR-managed pool used for retained or
  fetched KV.
- **G3 storage** is optional bounded file-backed storage.
- **Remote framework DRAM** is request-scoped peer memory identified by a
  router hint. KVCR does not maintain a global peer inventory.

A block can be resident in more than one tier while a copy is in flight.
Claims prevent a KVCR-owned residency from being evicted; framework pins keep
framework-owned sources valid. Operation state and residency state are
separate so cancellation of one caller does not invalidate resources still
used by another.

---

## Set up

### Prepare the standalone workspace

Run the standalone workflow from the root of an existing KVCR source checkout.
Before installing anything, confirm that the shell points to the intended
checkout and tools:

```bash
test -f pyproject.toml
uv --version
python3 --version
```

If `VIRTUAL_ENV` refers to an unrelated project, deactivate it before
continuing. KVCR uses its own `.venv`; later commands use `uv run` or name that
interpreter explicitly so packages are not installed into an outer environment.

The distribution name is `nvidia-kvcr`, the Python import is `kvcr`, and source
code lives under `src/kvcr`.

---

## Build and install

### Editable KVCR installation

The normal development installation is created by:

```bash
uv sync
```

Use this for standalone development. It preserves fast iteration and keeps the
dependency set described by `pyproject.toml`.

### KVCR wheel

Build a wheel when changing package metadata or validating the distributable
artifact:

```bash
uv build --wheel
```

The artifact is written under `dist/`, for example:

```text
dist/nvidia_kvcr-0.1.0-py3-none-any.whl
```

To replace the editable install temporarily without resolving a second
dependency closure:

```bash
uv pip install \
  --python .venv/bin/python \
  --reinstall \
  --no-deps \
  dist/nvidia_kvcr-*.whl
```

Use `--no-deps` only after `uv sync` has installed the declared dependencies.
Restore editable development mode after the wheel check:

```bash
uv sync
```

---

## Verify the installation

### Verify standalone editable provenance

Run from the KVCR checkout:

```bash
uv run python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata

import kvcr
from kvcr import KVCR, KVCRBindings

path = Path(kvcr.__file__).resolve()
print("distribution:", metadata.version("nvidia-kvcr"))
print("module:", path)
print("public API:", KVCR, KVCRBindings)
assert "src/kvcr" in str(path), path
PY
```

Also verify dependency consistency:

```bash
uv pip check --python .venv/bin/python
uv tree
```

### Verify an installed wheel

After installing the wheel, use the environment Python directly so `uv run`
does not restore the editable project first:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata

import kvcr
from kvcr import KVCR, KVCRBindings

path = Path(kvcr.__file__).resolve()
print("distribution:", metadata.version("nvidia-kvcr"))
print("module:", path)
print("public API:", KVCR, KVCRBindings)
assert "site-packages" in str(path), path
PY
```

---

## Develop with KVCR

### Use the public API lifecycle correctly

KVCR is constructed by a framework adapter, which supplies runtime
configuration, backend memory descriptions, and callbacks:

```python
from kvcr import KVCR, KVCRBindings
from kvcr.config import KVCRBackendConfigs, KVCRConfig

runner = KVCR(
    config=KVCRConfig(nixl_agent_name="worker-0"),
    bindings=KVCRBindings(
        request_pin=request_pin,
        poll_pin_results=poll_pin_results,
        release_pin=release_pin,
    ),
    backend_configs=KVCRBackendConfigs(...),
)
```

The callback names above represent services implemented by the framework
adapter; they are not provided by KVCR itself.

The main calls are:

| API | Purpose |
| --- | --- |
| `submit_hint()` / `discard_hint()` | Install and remove request-scoped router source information |
| `query()` | Read current local knowledge without blocking on the router or a transfer |
| `deposit()` | Copy framework-owned data into KVCR-managed storage |
| `fetch()` | Acquire data into KVCR-managed storage and return a releasable claim |
| `deliver()` | Place data into framework-provided destination descriptors |
| `poll_completed()` | Drain asynchronous per-entry outcomes |
| `release()` | Release KVCR residency claims returned by fetch or no-evict deposit |
| `abort()` | Best-effort cancellation of an operation or selected entries |
| `get_stats()` | Return a telemetry snapshot when telemetry is enabled |
| `close()` | Drain and synchronously tear down the runtime |

`query()` reports current knowledge rather than reserving data. A `HIT` can be
evicted before a later operation claims it, and a hinted remote source can
disappear. Callers must wait for the corresponding asynchronous completion.

### Development loop

Use a short feedback cycle:

1. Identify the module that owns the behavior.
2. Add or update a focused test for the expected terminal outcome.
3. Make the smallest implementation change.
4. Run the focused test while iterating.
5. Run the complete standalone validation before finishing.
6. If an adapter or router contract changed, run the optional integration
   validation separately.

Keep mechanism and policy separate:

- Transfers, pins, resource ownership, timeouts, and completion belong to mechanism.
- Admission, retention, placement, and eviction decisions belong to policy.

Policy calls must be quick and non-blocking. Event-loop code must not perform blocking external work.

---

## Validate standalone KVCR

### Standalone unit tests

Run a focused test while changing one subsystem:

```bash
uv run pytest tests/unit/test_progress.py -q
uv run pytest tests/unit/test_kvcr_service.py -q
```

Other useful selections are:

```bash
uv run pytest -k transfer -q
uv run pytest -x -vv
```

Run the complete framework-neutral suite before considering a KVCR-only change
validated:

```bash
uv run pytest -q
```

### Code quality

Run the configured checks across the checkout:

```bash
uv run ruff check .
uv run ruff format --check .
```

Apply formatting when needed:

```bash
uv run ruff format .
```

Keep public APIs typed and small. Prefer explicit configuration over hidden
process state. Errors should identify the operation and resource involved.
Telemetry labels must use bounded categories rather than block keys, request
IDs, or raw endpoints.

### KVCR service daemon

The KVCR service daemon owns pool lifecycle. It pre-allocates `--pool-count`
data-only pools before exposing its socket. A worker claims a pool by index; the
pool outlives that worker but not the service:

```bash
python -m kvcr.kvcr_service \
  --socket-path /run/kvcr/memory.sock \
  --pool-dir /dev/shm/kvcr \
  --pool-count 1 \
  --pool-size-gb 64 \
  --compatibility-digest example-model-layout
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--socket-path` | *(required)* | Unix socket the workers connect to |
| `--pool-dir` | *(required)* | Writable directory holding the pool files |
| `--pool-count` | *(required)* | Number of pools available by index |
| `--pool-size-gb` | *(required)* | Data-only size of each pool |
| `--compatibility-digest` | *(required)* | Exact digest every claimant must provide |

The pre-release wire protocol remains version 1. A worker calls
`KVCRClient.claim(pool_index, row_stride, compatibility_digest)`. The digest must
match the service exactly. The first successful claim fixes the service's row
stride; later claims with a different stride are rejected. The returned
`KVCRPoolHold` contains the mapped local-DRAM description and owns an exclusive
lease on that pool.

The service grants a pool to one live claimant at a time, and pool mappings are
not inherited by forked children. The `KVCRPoolHold` remains owned by the
claiming process and must not be used by a forked child. A second claim is
rejected while the claimant's pidfd reports it alive. The lease socket is
close-on-exec, and the service continues fencing the pool by that pidfd until the
process exits. Closing the claim connection, including an EOF, does not release a
live claimant's lease. `KVCRPoolHold.release()` first unmaps the pool locally,
then explicitly releases the lease and waits for the service's acknowledgement.
If the claimant exits without releasing, the service observes its pidfd and
makes the pool claimable again.

The same segment is reused across claims while the service runs, but this slice
does not recover cache records, so every claimant starts cold. Service shutdown
closes and removes all of that service's pool files. On startup, the service
also reclaims files orphaned by a crashed service; file locks prevent it from
removing pools that a live service or attached worker still uses.

Run the focused service tests after changing this subsystem:

```bash
uv run pytest \
  tests/unit/test_memory.py \
  tests/unit/test_kvcr_service.py \
  tests/unit/test_kvcr_service_workflow.py \
  -q
```

### Telemetry validation

Enable telemetry in `KVCRConfig` and provide a framework-specific
`stats_factory` through `KVCRBindings`. `get_stats()` should then expose
bounded counters, gauges, and duration observations.

The package exports metric definitions including `DURATION_METRIC`,
`TRANSFER_BLOCKS_METRIC`, `TRANSFER_BYTES_METRIC`, and `STATE_METRIC`.
Framework wrappers decide how those snapshots are mapped into their metrics
system.

Validate that counters increase on both success and failure paths, byte counts
agree with block geometry, timers use seconds consistently, and disabling
telemetry leaves the runtime behavior unchanged.

---

## Integrate with vLLM and Dynamo (optional)

Complete the standalone setup and validation first so framework, router, and native-runtime
failures are not confused with KVCR core failures.

### Build and install Dynamo

KVCR standalone tests do not require Dynamo. Install Dynamo when changing the
router-hint contract or running cross-worker integration tests. Refer to [Dynamo document](https://github.com/ai-dynamo/dynamo#building-from-source) for more detailed instructions.

For a standalone Dynamo checkout with its own environment:

```bash
uv venv --python 3.12 .venv
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install pip 'maturin[patchelf]'

cd lib/bindings/python
maturin develop --uv
cd ../../..

uv pip install -e .
```

This builds the Rust/Python bindings and installs the Dynamo Python packages.
Install backend extras only when needed. In particular, a Dynamo vLLM extra
may install its own released vLLM dependency and replace a customized editable
vLLM checkout.

### Build a shared Dynamo, vLLM, and KVCR integration environment

Use one shared virtual environment only for integration testing. Keep the
standalone KVCR `.venv` separate so framework dependencies cannot obscure
standalone failures.

A practical workspace is:

```text
integration-workspace/
├── .venv/
├── dynamo/
├── vllm/
└── kvcr/
```

Build in this order:

1. **Dynamo first.** Build its Rust bindings and install its Python package and
   required backend extras.
2. **vLLM second.** Install the compatible customized vLLM checkout in editable
   mode. This restores the intended source tree if a Dynamo extra installed a
   released vLLM package.
3. **KVCR last.** Install the current checkout in editable mode into the same
   interpreter used by vLLM workers.
4. **Reconcile native packages.** Verify PyTorch/CUDA, FlashInfer, vLLM native
   extensions, and NIXL after all resolver operations have completed.

Example commands, run from the integration workspace:

```bash
uv venv --python 3.12 .venv
export VIRTUAL_ENV="$PWD/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
uv pip install pip 'maturin[patchelf]' pytest

cd dynamo/lib/bindings/python
maturin develop --uv
cd ../../..
uv pip install -e .
uv pip install -e '.[vllm]'
cd ..

VLLM_USE_PRECOMPILED=1 \
  uv pip install --editable ./vllm --torch-backend=auto

uv pip install --editable ./kvcr
```

The precompiled vLLM path is valid only when the checkout and available wheel
artifacts are compatible. A customized branch may require an explicit wheel
commit/variant or a native build. Do not silently use the newest unrelated
wheel. Select the closest compatible artifact for the source revision, or use
the branch's documented native build.

The reference workspace scripts use the same ordering and then verify import
provenance and the complete native-runtime matrix. Their exact CUDA, PyTorch,
FlashInfer, and vLLM pins are examples for that workspace, not universal KVCR
requirements.

### Verify Dynamo

From the Dynamo checkout, using the environment into which Dynamo was built:

```bash
.venv/bin/python -m dynamo.frontend --help
```

The command should print frontend help and exit successfully. For a shared
integration environment, also verify the vLLM adapter import:

```bash
python - <<'PY'
import dynamo.vllm
print("dynamo.vllm:", dynamo.vllm.__file__)
PY
```

### Verify a shared integration environment

Run the following after Dynamo, vLLM, and KVCR are all installed:

```bash
python - <<'PY'
from pathlib import Path
import importlib
import importlib.metadata as metadata

modules = ["dynamo.vllm", "vllm", "kvcr", "nixl"]
for name in modules:
    module = importlib.import_module(name)
    print(f"{name}: {Path(module.__file__).resolve()}")

for distribution in ["nvidia-kvcr", "vllm", "nixl"]:
    print(f"{distribution}=={metadata.version(distribution)}")

import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is not visible to the integration env"
PY

uv pip check --python "$VIRTUAL_ENV/bin/python"
```

Confirm that `dynamo.vllm`, `vllm`, and `kvcr` resolve to the intended source
checkouts. A successful import from an unintended `site-packages` copy is a
provenance failure, even if its version string looks plausible.

If the selected vLLM build provides a native extension, import that extension
before running an end-to-end test. The extension name varies across vLLM
revisions; use the name expected by that checkout's own verification or tests.

### Configure and validate the integration

Keep this validation separate from the standalone development loop. Run it
when changing public KVCR contracts, the vLLM tier adapter, router hints,
control endpoints, block-key translation, framework pinning, or remote
transfers.

#### Minimal vLLM configuration

On a compatible vLLM revision, KVCR is a secondary tier behind
`OffloadingConnector` and `TieringOffloadingSpec`. The following example is for
one local DP rank and uses illustrative capacities and ports:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 1073741824,
    "enable_external_pinning": true,
    "self_describing_kv_events": true,
    "secondary_tiers": [
      {
        "type": "kvcr",
        "router_capabilities": ["router_hint"],
        "control_host": "0.0.0.0",
        "control_ports": [23280],
        "control_advertise_host": "127.0.0.1",
        "eager_ctrl_connect": true,
        "operation_timeout_ms": 1000,
        "enable_telemetry": true
      }
    ]
  }
}
```

Pass the serialized object through vLLM's `--kv-transfer-config` option. KV
events must also be enabled so Dynamo can maintain its cache inventory. A
minimal ZMQ publisher configuration is:

```json
{
  "publisher": "zmq",
  "topic": "kv-events",
  "endpoint": "tcp://*:23080",
  "enable_kv_cache_events": true
}
```

Pass that serialized object through `--kv-events-config`.

The important fields are:

| Field | Meaning |
| --- | --- |
| `cpu_bytes_to_use` | Capacity of vLLM's primary host-pinned offload tier, not an additional KVCR pool |
| `enable_external_pinning` | Allows KVCR to serve framework-owned host blocks while vLLM holds the required pins |
| `self_describing_kv_events` | Includes enough metadata for the router to interpret published KV events |
| `type="kvcr"` | Selects the KVCR secondary-tier manager |
| `router_capabilities` | Opts the tier into Dynamo router-hint source and destination planning |
| `control_host` | Local peer-control bind address |
| `control_ports` | One local control port for each DP rank managed by this worker |
| `control_advertise_host` | Host or address placed in worker registration and sent to peers |
| `eager_ctrl_connect` | Establishes peer control earlier; disabling it moves setup onto the request path |
| `operation_timeout_ms` | Deadline for KVCR operations; timeout begins safe cancellation and cleanup |
| `enable_telemetry` | Publishes KVCR operation, transfer, and state metrics through the vLLM wrapper |

For several local DP ranks, provide one `control_ports` entry per local rank in
local-rank order. Every worker must advertise an address reachable from its
peers, and every port must be unique on that host.

The secondary tier can optionally own local G2 capacity through
`secondary_g2_slots`, attach to a service-owned pool through
`kvcr_memory_server_socket`, or configure file-backed storage through `g3`.
Do not enable all capacity mechanisms blindly: a memory-service pool takes
precedence over an in-process `secondary_g2_slots` allocation. Policy names and
diagnostic options must match the KVCR and vLLM revisions being tested.

This example enables the worker side of the contract. Dynamo must still run a
KV-aware router that consumes KV events, selects a source and destination, and
places the resulting plan in the request metadata. A plain round-robin router
does not create KVCR source hints.

Use this progression so failures are localized:

1. **vLLM configuration and adapter tests** — no live router or transfer.
2. **Dynamo router-hint tests** — verify capability and endpoint publication.
3. **Two-worker transfer correctness** — exercise router, workers, peer
   control, NIXL, and output correctness together.
4. **Performance regression tests** — only after correctness is green.

---

## Troubleshooting

### KVCR cannot be imported

Confirm the interpreter and import path:

```bash
uv run python - <<'PY'
import sys
from pathlib import Path
import kvcr

print("python:", sys.executable)
print("kvcr:", Path(kvcr.__file__).resolve())
PY
```

If import fails, rerun `uv sync`. Confirm that the command uses the intended
`.venv`. The import is `kvcr`, and the distribution queried through package
metadata is `nvidia-kvcr`.

### Dependency or NIXL conflict

Inspect the resolved environment:

```bash
uv tree
uv pip check --python .venv/bin/python
```

Do not override the NIXL version declared by this checkout. If dependency
metadata changed, rerun `uv sync` rather than mutating individual packages
until the environment happens to import.

### The shared integration environment imports the wrong vLLM

Print import provenance:

```bash
python - <<'PY'
from pathlib import Path
import dynamo.vllm
import vllm
import kvcr

for module in [dynamo.vllm, vllm, kvcr]:
    print(module.__name__, Path(module.__file__).resolve())
PY
```

If vLLM resolves to an unintended released package, reinstall the customized
vLLM checkout after Dynamo and its extras, then reinstall KVCR. This is why the
integration build order is Dynamo → vLLM → KVCR.

### Native vLLM, PyTorch, CUDA, or FlashInfer mismatch

Record the relevant versions before changing packages:

```bash
python - <<'PY'
import importlib.metadata as metadata
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
for name in ["vllm", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"]:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}: not installed")
PY
```

Do not repair only the package named in the first import error. Reconcile the
entire native matrix, including the vLLM source revision and its precompiled
artifact, PyTorch's CUDA build, FlashInfer Python/cubin/JIT packages, and the
host driver.

### Dynamo does not produce router hints

Check the contract in this order:

1. The worker uses a role supported by the router-hint integration.
2. Exactly one secondary tier advertises `router_hint` capability.
3. `control_advertise_host` is present and reachable from other workers.
4. `control_ports` contains one valid port per local DP rank.
5. The worker registration visible to Dynamo includes the capability, role,
   and global-rank-to-endpoint mapping.
6. KV events reach the router and the source worker has overlap for the exact
   block-key namespace used by the request.

Do not infer hint delivery solely from a high router overlap score. Record the
hint payload at the framework boundary and verify that `submit_hint()` receives
the expected source endpoint, request ID, mode, and keys.

### Peer control connection fails

Distinguish bind and advertised endpoints. Binding to `0.0.0.0` does not make
`0.0.0.0` a usable peer destination. Confirm that:

- the source listens on the configured control port;
- the advertised address resolves from the destination process;
- no two ranks or test instances reuse a port;
- endpoint metadata uses the expected `tcp://host:port` form; and
- source and destination use compatible control-protocol metadata.

Enable eager control connection when isolating startup/handshake failures. Test
lazy connection separately because its cost and failures occur on the request
path.

### NIXL transfer fails or delivers incorrect data

Capture, for each operation:

- source and destination memory type, address range, length, and registration;
- block key and bytes per block;
- peer connection and metadata-exchange outcome;
- submit, active-transfer, and completion status;
- framework pin acquisition and release; and
- source and destination checksums in a test environment.

A successful control acknowledgement does not prove payload correctness. A
successful NIXL submission does not prove completion. Wait for the terminal
completion and verify the destination before exposing it to the framework.

### Remote delivery is partial

Partial delivery is expected when only part of a prefix remains readable. The
integration should consume the longest valid delivered prefix, recompute the
missing suffix, and preserve output correctness. Diagnose reductions at each
stage: router-planned blocks, source-resident blocks, pinned blocks, submitted
blocks, completed blocks, and destination-published blocks.

### Operations time out or fail slowly

Use telemetry to separate:

- hint age;
- peer setup;
- metadata/control handling;
- framework pin wait;
- NIXL submission;
- active transfer; and
- post-transfer notification.

These timers can overlap and should not be summed blindly. A long source-write
lifecycle with a short active-transfer timer usually indicates control,
pinning, scheduling, or stale-work overhead rather than insufficient transport
bandwidth.

### A test hangs or leaks resources

Run the smallest reproducer with detailed output:

```bash
uv run pytest path/to/test_file.py::test_name -vv -s
```

Check that every operation reaches a terminal state and that threads, sockets,
mapped memory, descriptors, claims, pins, temporary files, and child processes
are released on success, failure, timeout, and cancellation. Physical memory
release may need to wait for NIXL quiescence even after caller-visible timeout.

### The KVCR service does not start

Verify that:

- the socket parent and pool directory exist and are writable;
- the pool directory has capacity for every pool plus 100 MiB metadata per
  pool;
- another process is not listening on the socket;
- `--pool-count` is at least one; and
- `--pool-size-gb` is positive and finite.

The service removes a stale socket only after confirming no live service is
listening. It refuses to replace a socket owned by another live service.

### Minimum diagnostic record

When asking another developer to reproduce a failure, include:

- Python, KVCR, NIXL, Dynamo, vLLM, PyTorch, and CUDA versions;
- import paths for `kvcr`, `vllm`, and `dynamo.vllm`;
- the complete connector/tier and router-hint configuration;
- worker role, DP rank range, bind endpoint, and advertised endpoint;
- the focused command that reproduces the issue;
- source, destination, router, and test logs with aligned timestamps;
- operation outcome and duration telemetry; and
- whether failure occurs in standalone tests, adapter tests, router tests, or
  only the full transfer path.

This information normally identifies whether the fault is packaging,
configuration, routing, peer control, pinning, data transfer, policy, or
cleanup before a full serving stack is involved.
