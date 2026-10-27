# Two-GPU vLLM CI

`docker-build.yml` builds the integration image (`Dockerfile.ci`: the selected vLLM
image plus Dynamo wheels and KVCR from the tested commit), publishes it to ECR and
calls `vllm-e2e.yml` with the image pinned by digest. That workflow runs two jobs on
the two-GPU runner:

| Job | What it verifies |
| --- | --- |
| `peer-transfer` | `tests/e2e/run_vllm_kvcr.py`: data-parallel 2, seed rank 0, retrieve on rank 1; positive cached tokens and successful transfer / `remote_deliver` telemetry |
| `phase1` | `tests/e2e/run_phase1.py --mode guard --cases TC-01,TC-02,TC-03,TC-04`: KVCR-disabled reference engine, then Guard-mode TC-01 (store/commit/reuse), TC-02 (local DRAM flow incl. partial prefix), TC-03 (hinted peer transfer), TC-04 (Guard takeover and replacement) with token comparison and telemetry reconciliation on every request |

Guard mode needs a runner kernel of 6.5 or newer (`SO_PEERPIDFD`); otherwise the
harness reports BLOCKED and the job fails. TC-05 (performance arms) and TC-06
(Guard recovery timing) are not in the gate; run them manually or in a nightly
lane with `--cases TC-05,TC-06`.

## Running the same checks on a cluster

Inside any image that has KVCR, vLLM, Dynamo and NIXL installed, with two visible
GPUs:

```bash
python3 tests/e2e/run_phase1.py --output-dir artifacts/phase1/run --mode guard \
  --cases TC-01,TC-02,TC-03,TC-04
```

`scripts/docker/run_phase1.sh --gpus 0,1 --image <ref>` wraps the same run in a
container with the shared-memory and GPU settings the harness needs;
`scripts/k8s/phase1-pod.py` and `scripts/dlcluster/` cover Kubernetes and Slurm.
See `docs/test-plan-phase1.md` for every flag and the hybrid-model geometry.

## Evidence

`artifacts/phase1/run/` holds `result.json` (per-case checks, measurements,
limitations), `junit.xml`, `provenance.json` (package versions, harness digest,
image source revisions when present), `reference/` and `requests/` (every
request and response), and `stack/` (launch commands and logs of the frontend,
workers, `kvcr_service` instances and the state agent). A failed check names the
request stage and the metric values it compared; the matching request/response
files and worker logs are in the same tree.
