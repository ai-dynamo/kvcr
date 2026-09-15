# KVCR Developer Guide

This guide is for developers who are comfortable with Python, Linux, and
inference engines such as vLLM, but are new to KV Cache Runner (KVCR).
It covers the local development path from environment setup through
standalone validation.

To try the integrated stack without modifying source, follow the
[quick start](quick-start.md).

---

## Prerequisites

Use a Linux development environment with:

- Python 3.10 or newer;
- [`uv`](https://docs.astral.sh/uv/);
- a C/C++ runtime compatible with the NIXL wheel selected by the project;
- enough local memory and disk space for the tests you intend to run; and
- for service-backed recovery, in the guard service and every claimant: Linux 6.5
  or newer, and the system libatomic runtime (`libatomic1` on Debian and
  Ubuntu). Importing `kvcr` needs neither.

KVCR declares its Python dependencies in `pyproject.toml`. In particular, it
pins a compatible NIXL version. Let `uv` resolve that dependency instead of
installing a different NIXL release manually.

The framework-neutral unit suite does not require a GPU. Tests for a concrete
framework adapter, CUDA-aware NIXL transport, or cross-worker KV transfer may
require GPUs and the native dependencies of that framework.

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

The distribution name is `kvcr`, the Python import is `kvcr`, and source
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

### KVCR wheel (optional)

Build a wheel when changing package metadata or validating the distributable
artifact. After `uv sync` has installed the declared dependencies, build into a
fresh temporary directory and replace the editable install with that wheel:

```bash
KVCR_WHEEL_DIR=$(mktemp -d)
uv build --wheel --out-dir "$KVCR_WHEEL_DIR"
uv pip install \
  --python .venv/bin/python \
  --reinstall \
  --no-deps \
  "$KVCR_WHEEL_DIR"/kvcr-*.whl
```

Use `--no-deps` only after `uv sync` has installed the declared dependencies.
Next, [verify the installed wheel](#verify-an-installed-wheel) before restoring
editable development mode.

---

## Verify the installation

### Verify editable install

Run from the KVCR checkout:

```bash
uv run python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata

import kvcr
from kvcr import KVCR, KVCRBindings

path = Path(kvcr.__file__).resolve()
print("distribution:", metadata.version("kvcr"))
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

After installing the wheel, skip the editable provenance check above and use
the environment Python directly so `uv run` does not restore the editable
project first:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata

import kvcr
from kvcr import KVCR, KVCRBindings

path = Path(kvcr.__file__).resolve()
print("distribution:", metadata.version("kvcr"))
print("module:", path)
print("public API:", KVCR, KVCRBindings)
assert "site-packages" in str(path), path
PY
```

Restore editable development mode after the wheel check:

```bash
uv sync
```

---

## Develop with KVCR

### API lifecycle

KVCR is constructed by a framework adapter, which supplies runtime
configuration, backend memory descriptions, and callbacks:

```python
from kvcr import KVCR, KVCRBindings
from kvcr.config import KVCRBackendConfigs, KVCRConfig

runner = KVCR(
    config=KVCRConfig(
        nixl_agent_name="worker-0",
        pool_layouts=[("", block_size_bytes)],
    ),
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

For `on_resilience_event` and optional framework-memory quarantine, see the
[resilience contract](design_overview.md#failed-peers-and-dangling-operations).

See the design doc's [Framework–KVCR API](design_overview.md#frameworkkvcr-api)
and [operating flow](design_overview.md#operating-flow) for query, transfer,
completion, and release semantics, and the
[Router–KVCR API](design_overview.md#routerkvcr-api) for request-scoped hints.
`abort()` is currently unimplemented as it cannot be used by the frameworks.

### Development loop

Run a focused test while iterating, then the complete standalone validation
before finishing. If an adapter or router contract changed, run the optional
integration validation separately.

Keep policy decisions separate from transfer and resource management; see the
[Policy API](design_overview.md#policy-api). Policy calls and event-loop code
must remain non-blocking.

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

### KVCR guard service

The KVCR guard service owns pool lifecycle. It pre-allocates
`--guard-count` Guard-owned pool groups before exposing its socket. Every
group has the same ordered set of usable pool sizes from `--pool-sizes-gb`.
A worker claims a whole group by Guard index; its pools outlive that worker
but not the service.

Before starting, ensure `/run/kvcr` and `/dev/shm/kvcr` exist and are writable
by the user running the service:

```bash
uv run python -m kvcr.kvcr_service \
  --socket-path /run/kvcr/memory.sock \
  --pool-dir /dev/shm/kvcr \
  --guard-count 1 \
  --pool-sizes-gb 48,16 \
  --compatibility-digest example-model-layout
```

All flags below are required.

| Flag | Meaning |
| --- | --- |
| `--socket-path` | Unix socket the workers connect to |
| `--pool-dir` | Writable directory holding the pool files |
| `--guard-count` | Number of Guard-owned pool groups available by index |
| `--pool-sizes-gb` | Comma-separated usable sizes of the ordered pools in every group |
| `--compatibility-digest` | Exact digest every claimant must provide |

Each Guard gets one fixed 100 MiB recovery-journal region, added on top of the
listed usable sizes. The example therefore creates one mapping of 64 GiB plus
100 MiB, laid out as `[journal header + journal payload][pool 0][pool 1]`.
Pool sizes are rounded down to the native memory-page boundary.

A worker calls
`KVCRClient.claim(guard_index, pool_layouts, compatibility_digest, control_bind)`
with its Guard's control address and each ordered pool's name and block size.
The returned `KVCRPoolHold` owns the exclusive lease and exposes the mapped
pools through `local_dram.pools` as `(name, address, size_bytes)`. Pool block
sizes may differ. The pre-release wire protocol remains version 1.

`KVCRConfig.pool_layouts` supplies the same ordered layouts to direct and
`KVCRGuardConfig`-driven construction. Remote peers must match pool names,
block sizes, and order. G3 currently supports one pool and one block per key.

The compatibility digest must match the service and change whenever the KV
layout changes. A group's first claim fixes its ordered pool layout and, when
G3 is configured, its ordered paths, per-file capacity, backend and options,
and remote framework DRAM backend. Later mismatches are refused; changing the
layout requires restarting the service, which recreates the groups.

Pool mappings and the `KVCRPoolHold` must not be used by forked children;
create the shareable framework-control listener after the final fork.
The service fences each group by the claimant's pidfd until process exit.
Closing the claim socket, including across exec, does not release a live
claimant's lease. `KVCRPoolHold.release()` unmaps the group locally, explicitly
releases the lease, and waits for acknowledgement.

#### Crash recovery

A `KVCRGuardConfig` opts into a service pool group and its Guard together;
the framework control must support a shared listener or startup is refused.
Without this configuration, KVCR neither contacts the service nor builds a
Guard. Make the configured NIXL backends available in each process that uses
them; availability is not checked across processes.

The service binds the control endpoint and gives the claimant a duplicate.
After a worker dies, its Guard serves recovered G2 data from the whole pool
group at that same endpoint. A replacement primary takes back the group and
its recovered records; a claimant that cannot inherit the endpoint is refused.
A clean release returns the Guard to standby and the group to claimable.
Takeover and handback cost time linear in the number of recovered blocks.

Recovered blocks have no claims or old access timestamps and enter the
eviction list. Reuse gives them a new timestamp under the configured policy.
Recovery covers new requests; in-flight operations need caller-level retries.
The promoted Guard answers stale requests, even when it has no data to serve.

- Any Guard failure stops the service.
- Journal overflow disables warm recovery for that group. Watch for
  `KVCR pool recovery disabled`. The primary stops publishing and the Guard
  drops its mirrored state; the service continues. The journal is fixed at
  100 MiB, so larger blocks, shorter pool names, or fewer pool locations per
  key can reduce pressure.
- Currently, Guards do not serve G3. A replacement primary reopens it, but
  inherited G3 records are not verified against the files, which remain unlocked
  while the Guard serves. Sharing G3 paths is unsupported and undetected. Missing
  files are recreated with zeros, so deleting them can silently produce zero-filled
  cache hits. To discard a disk cache, restart the service.

Service shutdown closes and removes its pool files. Startup reclaims files
orphaned by a crashed service; file locks protect pools still used by a live
service or attached worker.

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
Each call returns the current interval snapshot and starts a fresh one.

The package exports metric definitions including `DURATION_METRIC`,
`TRANSFER_BLOCKS_METRIC`, `TRANSFER_BYTES_METRIC`, and `STATE_METRIC`.
Framework wrappers decide how those snapshots are mapped into their metrics
system.

Validate that counters increase on both success and failure paths, byte counts
agree with block geometry, timers use seconds consistently, and disabling
telemetry leaves the runtime behavior unchanged.

---

## Integrate with vLLM and Dynamo (optional)

Complete standalone setup and validation first. Use a separate shared environment
for Dynamo, vLLM, and KVCR to keep integration dependencies out of the standalone
development loop.

Select a vLLM revision that includes the KVCR secondary-tier adapter
(`"type": "kvcr"`, [PR #53624](https://github.com/vllm-project/vllm/pull/53624)), such as
[`a48bbcf`](https://github.com/vllm-project/vllm/commit/a48bbcfcdd2ac09cb729cf595026c5aec9b69ea0).
Pair it with a landed Dynamo revision such as
[`58f1e01`](https://github.com/ai-dynamo/dynamo/commit/58f1e01f76cbcf46a08963ffcab85c774da32a43),
which supports the versioned KV hint contract and one KVCR control port per
local data-parallel rank. The
[quick start](quick-start.md) records a pinned combination for its container.

### Build the integration environment

For details on Dynamo's Rust/Python build, see
[Building from source](https://github.com/ai-dynamo/dynamo#building-from-source).

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
2. **vLLM second.** Install the compatible vLLM checkout in editable
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

Precompiled vLLM artifacts must be compatible with the checkout. A customized
branch may require an explicit wheel commit/variant or its documented native
build. The quick start's CUDA, PyTorch, FlashInfer, and vLLM versions describe
that validated environment rather than universal KVCR requirements.

### Verify a shared integration environment

Run the following after Dynamo, vLLM, and KVCR are all installed. The first
command should print frontend help and exit successfully; the remaining
checks record import paths and native-package versions:

```bash
python -m dynamo.frontend --help

python - <<'PY'
from pathlib import Path
import importlib
import importlib.metadata as metadata
import sys

print("python:", sys.executable)
for distribution in [
    "kvcr", "vllm", "nixl", "torch", "flashinfer-python",
    "flashinfer-cubin", "flashinfer-jit-cache",
]:
    try:
        print(f"{distribution}=={metadata.version(distribution)}")
    except metadata.PackageNotFoundError:
        print(f"{distribution}: not installed")

import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)

modules = ["dynamo.vllm", "vllm", "kvcr", "nixl"]
for name in modules:
    module = importlib.import_module(name)
    print(f"{name}: {Path(module.__file__).resolve()}")

from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
print("KVCR secondary tier:", SecondaryTierFactory.get_tier_class({"type": "kvcr"}))

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

Run integration validation when changing public KVCR contracts, the vLLM tier
adapter, router hints, control endpoints, block-key translation, framework
pinning, or remote transfers.

#### Minimal vLLM configuration

On a compatible vLLM revision, KVCR is a secondary tier behind
`OffloadingConnector` and `TieringOffloadingSpec`. Use the quick start's
[worker configuration](quick-start.md#4-start-a-kvcr-enabled-worker) for the
`--kv-transfer-config` and `--kv-events-config` objects, adjusting capacities,
ports, and local DP ranks for your environment.

The important fields are:

| Field | Meaning |
| --- | --- |
| `cpu_bytes_to_use` | Capacity of vLLM's primary host-pinned offload tier, not an additional KVCR pool |
| `self_describing_kv_events` | Includes enough metadata for the router to interpret published KV events |
| `type="kvcr"` | Selects the KVCR secondary-tier manager |
| `router_capabilities` | Opts the tier into Dynamo router-hint source and destination planning |
| `control_host` | Local peer-control bind address |
| `control_ports` | One local control port for each DP rank managed by this worker |
| `control_advertise_host` | Host or address placed in worker registration and sent to peers |
| `eager_ctrl_connect` | Establishes peer control earlier; disabling it moves setup onto the request path |
| `local_dram_backend` | NIXL backend used for local DRAM transfers |
| `remote_fw_dram_backend` | NIXL backend used for peer DRAM transfers |
| `operation_timeout_ms` | Deadline for KVCR operations; timeout begins cancellation and cleanup |
| `abandon_timeout_ms` | Deadline from operation start to report unresolved memory as uncertain; default `5000`, at least twice `operation_timeout_ms` |
| `enable_telemetry` | Publishes KVCR operation, transfer, and state metrics through the vLLM wrapper |

For several local DP ranks, provide one `control_ports` entry per local rank in
local-rank order. Every worker must advertise an address reachable from its
peers, and every port must be unique on that host.

The secondary tier can optionally own local G2 capacity through
`secondary_g2_slots`, attach to a service-owned pool through
`kvcr_service_socket_path` together with `compatibility_digest`, or configure
file-backed storage through `g3`.
The vLLM adapter uses one unnamed pool per Guard, so start the guard service with
one pool size, for example `--pool-sizes-gb 48`.
Do not enable all capacity mechanisms blindly: a pool owned by the guard service takes
precedence over an in-process `secondary_g2_slots` allocation. Policy names and
diagnostic options must match the KVCR and vLLM revisions being tested.

The worker configuration enables its side of the contract. Dynamo must still run a
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

Use the [installation checks](#verify-the-installation) to confirm the
interpreter and import path.

If import fails, rerun `uv sync`. Confirm that the command uses the intended
`.venv`. The import is `kvcr`, and the distribution queried through package
metadata is `kvcr`.

### Dependency or NIXL conflict

Run the dependency checks under [Verify the installation](#verify-the-installation).

Do not override the NIXL version declared by this checkout. If dependency
metadata changed, rerun `uv sync` rather than mutating individual packages
until the environment happens to import.

### The shared integration environment imports the wrong vLLM

Use the [shared-environment checks](#verify-a-shared-integration-environment)
to print import paths.

If vLLM resolves to an unintended released package, reinstall the selected
vLLM checkout after Dynamo and its extras, then reinstall KVCR. This is why the
integration build order is Dynamo → vLLM → KVCR.

### Native vLLM, PyTorch, CUDA, or FlashInfer mismatch

Record versions with the
[shared-environment checks](#verify-a-shared-integration-environment) before
changing packages.

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
a protocol-conforming hint and the expected request ID.

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

### The KVCR guard service does not start

Verify that:

- the socket parent and pool directory exist and are writable;
- the pool directory has capacity for every Guard's full allocation: the sum
  of `--pool-sizes-gb` plus one 100 MiB journal. A group changing hands briefly
  appends its handback snapshot past that size;
  where there is no room for it, that handover comes back cold and the
  service carries on;
- another process is not listening on the socket;
- `--guard-count` is at least one; and
- every `--pool-sizes-gb` item is positive, finite, and at least one memory
  page.

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
