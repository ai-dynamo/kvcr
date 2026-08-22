# KVCR Extraction Provenance

## Immutable Sources

- Private-fork source commit:
  `b2df38fa2ce0487d380e668c02468812f17c7574`
- KVCR repository base:
  `e31766fcc7851117a1da7ba2c37d448aed05e057`

The pinned source commit exists only in NVIDIA's private, NVIDIA-visible-only
fork of vLLM. It was never in upstream vLLM. See [Authorship](#authorship) for
what that means for the extracted code.

## Authorship

All contributors to the extracted modules are NVIDIA employees.

The extracted modules are NVIDIA-authored. They were written in NVIDIA's
private fork of vLLM, which is visible only within NVIDIA, and were kept there
temporarily under `vllm/v1/kv_offload/tiering/kvcr/` before being moved to this
repository. They were never present in upstream vLLM. They have since been
removed from the private fork, so this repository is now their only home.

## Source Inventory

### Modules extracted from the private vLLM fork

Six production modules were copied from the pinned commit. Three of them were
later renamed or split, so the current tree maps back to the pinned source as
follows:

| Pinned path in the private fork | Current KVCR destination |
|---|---|
| `vllm/v1/kv_offload/tiering/kvcc/kvcc.py` | `src/kvcr/core.py`, and the parts split into `src/kvcr/api.py` and `src/kvcr/config.py` |
| `vllm/v1/kv_offload/tiering/kvcc/kvcc_types.py` | `src/kvcr/types.py`, and the parts split into `src/kvcr/api.py` and `src/kvcr/config.py` |
| `vllm/v1/kv_offload/tiering/kvcc/local_dram.py` | `src/kvcr/local_dram.py` |
| `vllm/v1/kv_offload/tiering/kvcc/peer_control_channel.py` | `src/kvcr/control_channels.py` |
| `vllm/v1/kv_offload/tiering/kvcc/progress.py` | `src/kvcr/progress.py` |
| `vllm/v1/kv_offload/tiering/kvcc/remote_fw_dram.py` | `src/kvcr/remote_fw_dram.py` |

`api.py` and `config.py` are not new code. They hold the public API surface
and the configuration and telemetry contracts that were relocated out of the
extracted `kvcc.py` and `kvcc_types.py` when the package was reorganized, plus
subsequent additions to those same surfaces. The table preserves the exact
paths in the immutable source commit while listing the current destinations.

### Modules written in this repository

All KVCR modules are NVIDIA-authored; this section separates them by where they
were first written, not by authorship. These modules never existed in the
private fork. They were written here after the extraction:

| KVCR module | Contents |
|---|---|
| `src/kvcr/policy.py` | Public placement and eviction policy API |
| `src/kvcr/policy_runtime.py` | Policy invocation and eviction candidate ordering |
| `src/kvcr/local_disk.py` | File-backed G3 residency and its NIXL transfers |
| `src/kvcr/memory.py` | Server-owned shared-memory pools |
| `src/kvcr/guard_protocol.py` | KVCR service wire protocol and client |
| `src/kvcr/kvcr_service.py` | Standalone Unix-socket KVCR service daemon |

`src/kvcr/__init__.py` is the package's re-export surface and was also written
here.

`manager.py`, factory registration, vLLM adapters, and vLLM integration and
end-to-end tests were not part of the framework-neutral extraction. The
planned quick-start image will fetch a public adapter revision after the
upcoming vLLM PR is available, as described below.

## Planned Quick-start vLLM Integration Overlay

The public E2E quick start depends on six vLLM integration files that are
planned for an upcoming public vLLM PR. Once that PR is available, the
container will fetch its immutable public commit and copy the files into the
pinned vLLM image at build time. They are not stored in this repository or
included in the `nvidia-kvcr` wheel.

- Planned source: upcoming public vLLM PR
- Repository placeholder: `PUBLIC_VLLM_PR_REPOSITORY_PENDING`
- Revision placeholder: `PUBLIC_VLLM_PR_COMMIT_PENDING`
- Upstream vLLM base: `6adad08767583f52eb4d2122111af0bf638ed5e6`
- Pinned published vLLM image revision:
  `8efa13b700f1836657699cae2503dc2feab27fa0`

The Dockerfile intentionally rejects the placeholder values until the public
PR exists. When it is available, this section must be updated with the exact
public repository, commit, and compatibility relationship. The prepared
overlay copies these paths:

| Fetched vLLM path |
|---|
| `vllm/distributed/kv_events.py` |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py` |
| `vllm/v1/kv_offload/base.py` |
| `vllm/v1/kv_offload/tiering/factory.py` |
| `vllm/v1/kv_offload/tiering/kvcr/__init__.py` |
| `vllm/v1/kv_offload/tiering/kvcr/manager.py` |

These files are expected to retain their Apache-2.0 contributor headers. The
four existing vLLM files contain upstream vLLM code plus the integration
changes; the two `tiering/kvcr` adapter files are added by the integration.

## Relationship to the Pinned Source

At the time of extraction the six copied modules were parity checked against
the pinned source after applying the normalizations below. That parity is
a statement about the extraction, not about the current tree: KVCR development
has continued in this repository since, and the extracted modules have been
modified by NVIDIA-authored changes such as the policy API, the G3 tier, the
NIXL transfer lifecycle rework, and telemetry additions.

The pinned commit therefore records where the extracted code came from. It is
no longer a parity target, and the repository is not maintained against it.

### Normalizations applied at extraction

The six copied modules differed from the pinned source only by:

1. replacing the vLLM repo-wide boilerplate file header with the approved
   NVIDIA notice;
2. updating relative imports after `kvcr_types.py` moves to `types.py`; and
3. replacing the vLLM NIXL utility import with a local compatibility module.

The header replaced in item 1 was the boilerplate the private fork applied to
every file in the vLLM tree regardless of that file's origin. It was not
attribution for upstream-derived code. The six extracted modules are wholly
NVIDIA-authored and contain no upstream vLLM code, so no third-party
attribution exists that Apache 2.0 would require to be retained alongside the
NVIDIA notice.

No behavioral changes were part of the extraction itself. The compatibility
module in item 3 was later removed once the vLLM-specific NIXL assumptions were
dropped; KVCR now uses the `nixl` package directly.

## Repository history

This repository was initialized with a fresh history from the internal
repository NVIDIA-dev/kvcc at commit `5fe3829f062befc21d7b7b05cc5031230a82e9ae` (2026-08-21). The
internal repository remains the historical archive; commits referenced
above are reachable there, not in this repository's history.
