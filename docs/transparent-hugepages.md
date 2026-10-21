# Transparent hugepages for Guard pools

KVCR-Service can opt its shared DRAM pools into transparent hugepages (THP).
The default remains the existing shared `mmap` path and the host's page policy.
It does not force base pages or disable THP globally.

Choose the mode with a JSON string when starting the service:

```bash
python -m kvcr.kvcr_service \
  --socket-path /tmp/kvcr-service.sock \
  --pool-dir /dev/shm \
  --guard-count 1 \
  --pool-sizes-gb 1 \
  --compatibility-digest my-model-layout \
  --memory-config '{"mode":"thp"}'
```

Use `--memory-config '{"mode":"default"}'`, or omit the option, for the
original behavior. `{}` also selects the default. Unknown fields and unsupported
modes fail validation. The mode selector allows future memory policies without
changing this JSON object's shape; this branch implements only `default` and
`thp`. HugeTLB and the experiment's base-prefault mode are not implemented here.

For Python service construction, decode the same string with
`msgspec.json.decode(value, type=KVCRMemoryConfig)` or construct
`KVCRMemoryConfig(mode="thp")` from `kvcr.config`, and pass it as `memory_config`
to `_KVCRService`. This service constructor remains an internal API.

The service owns the allocation policy. It records `memory_mode` in the pool
specification sent to workers, so the Guard and every claimant use the same
mapping policy without environment variables. Use the branch on both service
and worker processes: an older worker can ignore the new grant field and would
not apply the requested THP policy. Grants without that field default to
`default` when read by this branch. Framework-owned GPU buffers and externally
supplied local DRAM regions are unaffected.

## Mapping and registration order

```text
KVCR-Service initialization
    parse memory-config JSON
    validate total pool length against kernel PMD size for THP
    create pool file, reserve backing space, record mode in pool spec
    initialize Guard
        attach pool using mode from spec
            default: ordinary shared mmap + DONTFORK
            thp: reserve owned address range; align shared mapping to PMD size
                 DONTFORK; HUGEPAGE; POPULATE_WRITE; COLLAPSE
                 verify full shared THP backing in /proc/self/smaps
        initialize recovery journal

worker claims pool
    receive pool spec (including memory_mode)
    attach using the same policy; verify THP before returning attachment
    NIXL register_memory using the resulting pool addresses
    prepare transfer descriptors
    issue and complete individual transfers using the registered mapping
```

The Guard also registers its prepared attachment with NIXL when it serves the
pool. Population, collapse and backing verification happen on attachment, before
native pinning. There is no added per-transfer collapse or registration step.
The recovery snapshot tail still uses ordinary mappings. Pool and journal
geometry are preserved; the implementation does not silently add padding or
increase cache capacity.

The aligned mapping is ported from the tested implementation's THP path. It
supports Python's buffer protocol so the recovery journal can use it, refuses
unmapping while a buffer is exported, and releases its reservation on an
initialization failure. `MAP_FIXED` replaces only an owned reservation.

## Requirements and failure behavior

- THP mode requires Linux and Python 3.12 or later (PEP 688 buffer support).
  Default mode retains the package's Python version requirements.
- Use shared backing that supports THP, normally a sufficiently sized tmpfs
  such as `/dev/shm`. THP needs no reserved HugeTLB pool or hugetlbfs mount.
- `MADV_POPULATE_WRITE` and `MADV_COLLAPSE` must be supported and permitted.
  `MADV_COLLAPSE` was introduced in Linux 6.1; actual shared backing support
  must be verified on the deployed kernel. KVCR changes no host policy.
- The **total pool mapping length**, journal plus all cache pools for one Guard,
  must be a multiple of the kernel PMD hugepage size. The CLI's journal is
  100 MiB; the example's 1 GiB cache yields a 1124 MiB total, divisible by a
  typical x86 PMD size of 2 MiB. Address alignment is handled internally.
- PMD size is read from
  `/sys/kernel/mm/transparent_hugepage/hpage_pmd_size`, not hardcoded to 2 MiB.
  The sampled GB200 kernels used 64 KiB base pages and 512 MiB PMD pages; the
  example's geometry does not fit those hosts. Additionally, their shared-memfd
  collapse probe returned `EINVAL`. This branch does not claim to solve that
  host configuration's shared THP limitation.
- Any population/collapse error or incomplete backing aborts attachment. There
  is no fallback that could make a THP experiment unknowingly measure default
  pages. Startup therefore fails if a Guard cannot obtain its requested mapping.

Full backing means `ShmemPmdMapped + FilePmdMapped` equals the mapping length in
its smaps entries. `KernelPageSize` alone does not establish THP backing.
Verification is an attachment-time observation, not a guarantee against later
kernel changes, and THP capability does not establish a transfer speedup.

## Testing

Install the repository's locked development dependencies and run:

```bash
uv sync --frozen --group dev
uv run pytest tests/unit -q
uv run ruff check .
uv run ruff format --check .
```

The THP tests cover JSON validation, default compatibility, pool-grant mode
round trips, service-to-Guard propagation, actual aligned shared THP backing,
cross-process writes, journal access, exported-buffer lifetime, snapshots,
incomplete-backing cleanup, geometry rejection and the Grace PMD-size value.
Real mapping tests need a host that supports shared THP; they skip large PMD
sizes to keep unit-test allocations bounded. Cross-node transfer performance
must be measured separately on matching hosts and configurations.

Validation on 2026-10-21 with Python 3.12.3 and the locked NIXL 1.5.0:
545 unit tests passed; repository Ruff checks and formatting passed. A separate
native UCX smoke check registered an 8 MiB THP pool, verified all 8388608 bytes
of shared THP backing while registration was live, then deregistered and closed
it. This local check does not measure cross-node throughput or latency.
