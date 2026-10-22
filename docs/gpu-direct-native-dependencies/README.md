<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Experimental GPU transport dependencies

The GPU-only endpoint requires a native stack that accepts its explicit CUDA IPC
or MNNVL profile and detects CUDA allocations without enabling CUDA-copy transports.
The patches in this directory make those dependency changes reviewable. They
are experimental patches against exact public source commits, not released
NIXL/UCX support. Installing the KVCR Python wheel does not install these patches
or native binaries.

| Dependency | Base commit | Patch |
| --- | --- | --- |
| [NIXL](https://github.com/ai-dynamo/nixl) | `075b2397029fc67a0b69a34df566c1ae3d00f5ab` | [Strict CUDA IPC policy](nixl-cuda-ipc-only.patch) |
| [UCX](https://github.com/openucx/ucx) | `65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0` | [CUDA IPC memory detection](ucx-cuda-ipc-detection.patch) |

[source-lock.json](source-lock.json) records the patch hashes. NIXL changes use
its Apache-2.0 license; UCX changes retain the upstream BSD license in
[UCX-LICENSE.txt](UCX-LICENSE.txt). No third-party runtime library is bundled.

The MNNVL profile adds [nixl-mnnvl-only.patch](nixl-mnnvl-only.patch) after the
existing NIXL patch and
[ucx-mnnvl-fabric-detection.patch](ucx-mnnvl-fabric-detection.patch) after the
existing UCX patch. [mnnvl-source-lock.json](mnnvl-source-lock.json) records the
additional hashes and apply order against the same public bases. The original
CUDA IPC patch and lock remain unchanged.

## MNNVL backend policy

Create the dedicated UCX backend with `gpu_direct_transport="mnnvl"`,
`gpu_direct_ucx_tls="cuda_ipc,tcp,self"` and `ucx_error_handling_mode="none"`.
Both new parameters default to empty for ordinary backends. Actual backend
parameter readback must report the selected policy before VRAM registration.
The native MNNVL constructor forces these settings on its own UCP configuration:

| UCP setting | Value |
| --- | --- |
| TLS | cuda_ipc,tcp,self |
| CUDA_IPC_ENABLE_MNNVL | y |
| CUDA_IPC_GPU_DIRECT_FABRIC_ONLY | y |
| PROTO_ENABLE | y |
| PROTO_EMULATION_ENABLE | n |
| TCP_PUT_ENABLE | n |

TCP retains active-message control, while its native PUT capability and UCP's
software PUT/GET protocols are disabled. Only `cuda_ipc` supplies native CUDA
RMA in this closed transport set. Missing settings, unsupported fabric or an
unavailable route fail instead of enabling another GPU payload transport.
These settings do not mutate the process environment or ordinary P-to-D contexts.

MNNVL VRAM registrations must cover bounded physical DEVICE VMM allocations
created with CUDA FABRIC handles. Native checks reject legacy, managed, host and
out-of-bounds memory. The new per-MD FABRIC-only detector is default-off, so the
existing CUDA IPC detector retains its legacy-allocation rules even on MNNVL
hardware. Existing UCX FABRIC handle export, import, remote mapping and completion
are reused. A matching NVML fabric UUID/clique and an allocated common IMEX
channel are required; actual cross-host export/import must still succeed.

The NIXL plugin's new registration checks link the existing CUDA dependency.
Build with CUDA 12.4 or later. Rebuild the complete CUDA IPC module against the
modified private MD/config headers; its private struct layout changes. Record
actual source and loaded-library hashes rather than assuming binary identity
from an unchanged version string. These are experimental dependency patches,
not stock NIXL/UCX functionality supplied by the KVCR wheel.

Run the added native parameter, real FABRIC registration and default-versus-strict
MD tests on allocated hardware. Then require a two-host KVCR full-byte test with
distinct GPUs in one observed fabric, a job-reserved IMEX channel, untouched
guards, native pin/quiescence checks and an isolated CUDA activity review. A
same-host test or skipped hardware case cannot qualify cross-host MNNVL. Model
correctness and serving performance require their own complete runs.

## Native behavior

With `NIXL_UCX_CUDA_IPC_ONLY=1`, NIXL requires an explicit `UCX_TLS` containing
`cuda_ipc` and only optional `posix`, `sysv` and `self` control transports. It
rejects aliases, duplicates, network transports and CUDA/GDR copy transports,
then binds the validated list to the actual UCP configuration. The initialized
context must still advertise CUDA memory support.

The same strict profile defaults the advertised and actual UCX endpoint mode
to `none`, including ordinary eager SDK callers that cannot supply backend
parameters. It records the effective mode and rejects explicit `peer` under
this profile. The flag-unset default remains `peer`; ordinary explicit endpoint
options retain their behavior. Shared-memory active-message transports do not
support native peer-error handling. A deadline does not cancel native writes,
and uncertain source/target memory must remain retained until actual quiescence.

The UCX correction advertises CUDA detection in the CUDA IPC memory domain and
uses real CUDA driver pointer attributes and allocation bounds. It accepts
legacy IPC-capable device allocations, including valid interior ranges. It
rejects host, managed, unsupported, zero-length and out-of-bounds ranges. It
does not add payload copies or enable the CUDA-copy memory domain.

## Build and qualify

Apply each patch to its exact base with `git apply --check` followed by
`git apply`. Build UCX and NIXL using their pinned source build instructions:
[UCX source build](https://github.com/openucx/ucx/blob/65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0/README.md)
and [NIXL source build](https://github.com/ai-dynamo/nixl/blob/075b2397029fc67a0b69a34df566c1ae3d00f5ab/README.md).
NIXL's `ucx_path` Meson option must reference the matching UCX installation.
Keep CUDA version, architecture, UCX headers, feature macros and native core
ABI consistent. Replacing a library in an existing wheel requires separate
ABI and actual-loaded-library validation; a version string is insufficient.

Set the strict policy before importing or constructing native agents:

```bash
export NIXL_UCX_CUDA_IPC_ONLY=1
export UCX_TLS=cuda_ipc,posix,self
```

For a build that packages separate UCX modules, ensure exactly one intended
CUDA module is loaded. `UCX_MODULES=cuda` can exclude unused IB modules in an
isolated IPC-only qualification. It does not replace validation of the actual
installed libraries or qualify a different transport profile.

Run the added NIXL option/TLS tests and UCX CUDA IPC memory-domain tests on the
native build. Before a model test, require actual VRAM registration, ordinary
eager SDK construction without an explicit endpoint override, and the
[two-peer KVCR byte test](../../tests/integration/gpu_direct_roundtrip.py).
Verify full payload and untouched guard rows, pin lifetime and quiescent close.
Independently review CUDA activity over the payload window: peer-copy bytes
must cover the expected payload, with no host staging. An empty trace, endpoint
label, successful constructor or source-only test does not establish that claim.
Separate-container GPU exposure requires an additional deployment test.

Native construction, isolated transport qualification, full model correctness
and end-to-end performance are separate results. Preserve the exact source,
library hashes, workload and failed attempts for each.
