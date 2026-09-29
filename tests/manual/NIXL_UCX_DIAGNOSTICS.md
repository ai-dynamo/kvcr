# NIXL / UCX self-copy diagnosis

All payload copies in this probe use the same NIXL agent and its UCX backend.
There is no alternate copy engine. It verifies bytes in both GPU-to-pinned-DRAM
and pinned-DRAM-to-GPU directions. Registration and warmup are outside the
reported samples; samples include transfer preparation, submission and polling,
not only DMA time. Run without concurrent model work for timing comparisons.

```bash
export NIXL_LOG_LEVEL=DEBUG
export UCX_LOG_LEVEL=DEBUG
export UCX_PROTO_INFO=y
export UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self
python tests/manual/nixl_ucx_self_copy.py --threads 4 --error-mode peer \
  --report self-peer.json >self-peer.log 2>&1
```

Use a fresh process for each case. Compare `--threads 0` versus `4`, and
`--error-mode peer` versus `none`. Include small and large descriptor lists:
`--spans 64 --span-bytes 131072`, `--spans 2048 --span-bytes 4096`, and
`--spans 1 --span-bytes 8388608`. NIXL's worker-pool splitting threshold is
version-dependent, so one large buffer is not equivalent to many small spans.

Read the protocol table for the **actual memory pair and endpoint**, e.g.
`ucp_put*(multi) from cuda/GPU0 to host`, not just the list of available
transports. Internal UCX memory-type endpoints can use `cuda_copy` even when
the application's self-copy endpoint uses `rc_mlx5`.

Additional diagnostic controls (not production recommendations):

- `UCX_TLS=cuda_copy,self --threads 0 --error-mode none` removes network
  transports from a local-only probe. Do not apply it to a two-node server.
- `UCX_CUDA_COPY_BW=300GBs,h2d:300GBs,d2h:300GBs` changes the selection cost
  model, **not** measured bandwidth. Check the selected protocol afterwards.
- Repeat without DEBUG / protocol logging for timing. Debug I/O perturbs
  scheduling, and a faster isolated copy does not establish lower model TTFT.

`peer` error handling and CUDA-interface reachability are separate constraints.
Disabling peer-error handling globally weakens remote failure behavior; this
probe intentionally does not change KVCR's production defaults. A same-agent
transfer is not necessarily a same-UCX-worker/interface transfer.

In KVCR itself, set `LocalDramOptions(device_copy=False)` to exercise NIXL
self-copies. With `NIXL_LOG_LEVEL=DEBUG`, KVCR logs its agent/thread setup and
each submitted memory pair, descriptor count, and byte count. The backend in
that line is the **requested** backend, not proof of UCX's selected transport.
No addresses or key payloads are included. Native UCX protocol logs provide the
selection evidence.

## Representative descriptor geometry

Large total payloads do not necessarily mean large individual copies. Replay
the framework's descriptor-size histogram, not only its total byte count:

```bash
python tests/manual/nixl_ucx_self_copy.py --threads 0 --error-mode none \
  --span-pattern 8704:6,17408:2,74880:6,149760:2 --gap-bytes 4096 \
  --warmup 5 --iterations 30 --report mixed-spans.json
```

That example transfers 835840 payload bytes in 16 spans. A larger mixed case is
`8704:3,17408:1,74880:3,149760:41` (6408320 bytes in 48 spans). Gaps model
non-contiguity only; they do not reconstruct actual layer/pool placement.
Random byte payloads are checked only over the transferred spans.

For a matched local-path comparison, keep the worker count, error mode, CPU/GPU
affinity, host NUMA placement, and geometry fixed. Confirm actual UCX protocols
with logging first; then time quiet runs, alternating case order across fresh
processes. The CUDA bandwidth-model override above is a diagnostic selection
control. Keep the production four-thread/peer-error mode as a separate baseline.

Sweep both total size and per-span size. For example, at 128 MiB total compare
`--spans 1024 --span-bytes 131072` with `--spans 128 --span-bytes 1048576` and
`--spans 1 --span-bytes 134217728`. A coalesced buffer is a performance control,
not proof that scattered model pages can be merged without packing or layout
changes. Do not infer end-to-end offload or request latency from this probe.
