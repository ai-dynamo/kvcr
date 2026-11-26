# Two-GPU vLLM CI

`docker-build.yml` builds `Dockerfile.ci` from the selected vLLM image, installs
Dynamo and KVCR from the tested commit, publishes to ECR, and passes the image
by digest to `vllm-e2e.yml`. Manual builds can select the base image, Dynamo
version and KVCR ref. A newer run on the same branch cancels the older run.

## Coverage

Each model runs aggregated and disaggregated, in native and Guard modes:

| Model | Architecture |
| --- | --- |
| `Qwen/Qwen3-0.6B` | Dense attention |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | Hybrid attention/Mamba |
| `Qwen/Qwen3.5-0.8B` | Hybrid attention/GDN |

Pinned model revisions and launch settings are in `tests/e2e/ci_profiles.json`.
Nemotron uses an FP32 output head to reduce numerical sensitivity in exact
reference comparisons; the model remains configured as BF16.
TC-01–03 run in all 12 configurations; TC-04 runs in the six Guard configurations.
Native means KVCR without Guard. Outputs are compared exactly with an independent
KVCR-disabled reference.

| Case | Checks |
| --- | --- |
| TC-01 | Cold output, KVCR commit and warm reuse. |
| TC-02 | Prefix delivery, misses and release; partial-hit assertions apply when the reference engine supports that reuse. |
| TC-03 | Hinted peer transfer with source/target KVCR telemetry, plus a separate cold request to check hint isolation. |
| TC-04 | Guard takeover after worker death, preserved-prefix transfer, replacement attachment and cache-hit/output checks. |

## Execution

Each job owns two GPUs, with TP=1/DP=1 per worker. Aggregated workers use GPUs
`0,1`; disaggregated uses two prefill workers on GPU `0` and a decoder on GPU `1`.
Jobs run independently: a failure does not cancel the other configurations.
Available runner capacity determines parallelism. Guard requires Linux 6.5+.

The profile runner prefetches the pinned model into the standard Hugging Face
cache before starting workers. CI uses `/scratch/huggingface`. The harness
manages its workers, services and temporary directories.

Inside the integration image with two visible GPUs:

```bash
python3 -u tests/e2e/run_ci_profile.py \
  --profile vllm-mamba-disagg-guard --output-dir artifacts/mamba-disagg-guard
```

Add `--dry-run` to print the command without running tests, or `--gpus` to override
local placement. Use a fresh output directory for each run. Add model settings in
`ci_profiles.json`; test assertions live in
[`kvcr_e2e/cases.py`](../tests/e2e/kvcr_e2e/cases.py).

## Results

Failed, blocked or missing cases fail CI. Each `<profile>-<attempt>` artifact
contains `run/result.json`, `run/junit.xml`, `run/provenance.json`, reference and
request/response records, and `run/stack/` launch commands, logs and cleanup
records. Model download output is in `prefetch.log`.
