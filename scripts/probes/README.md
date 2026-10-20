# Bare-vLLM probes

Standalone scripts run against `vllm serve` (no KVCR, no Dynamo) to establish
engine behaviour the Phase 1 verdicts rely on. Qwen3.5-4B, `--mamba-cache-mode align`.

| script | question | finding (2026-10-18/19) |
| --- | --- | --- |
| `ssm_prefix_probe.py` | does a prompt that is a strict prefix of an earlier one hit? | no: one SSM checkpoint per request at the last 576-boundary <= prompt end |
| `ssm_fork_probe.py` | does the first fork from a cached prefix leave a junction checkpoint? | yes: first fork 0, later siblings hit the block-floored junction |
| `ssm_events_probe.py <label>` | what does the offload tracker publish per KV cache group? | before vllm-project/vllm#62613 SSM groups publish `block_size=0` placeholders; after, `block_size=576`, `kind=mamba` |

Hits are read from `vllm:prefix_cache_hits_total` deltas because bare vLLM
completions do not fill `usage.cached_tokens`. Events are read from the ZMQ
publisher with `msgspec`.
