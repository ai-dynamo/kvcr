#!/bin/bash
#SBATCH --job-name=doxu-kvcr-phase1
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --constraint="2GPU&x86_64&UB24.04"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/home/doxu/kvcr-runs/%j.out
#
# KVCR test plan Phase 1 (TC-01..TC-06) on one dlcluster node with 2x H100.
# submit.sh adds --exclusive for the 2-GPU request: shared 8-GPU nodes hand out
# GPUs with other tenants' memory on them and vLLM's 0.8 utilization check then
# fails at startup. With --gpus 1 (submit.sh) both workers share one GPU at
# KVCR_GPU_UTIL (default 0.35), which tolerates a shared node.
# x86_64: the linux/amd64 image build segfaults under QEMU on aarch64 nodes.
# UB24.04: kernel 6.8 (Guard needs >= 6.5); the UB22.04 nodes run 5.15 and the
# ones seen so far also cannot reach github.com for the source resolve step.
# Inputs (env, set by submit.sh):
#   KVCR_BUNDLE   git bundle of the kvcr-phase1-ci branch (in $HOME)
#   KVCR_REF      commit to check out from the bundle
#   KVCR_MATRIX   space-separated config names to run (default: dense hybrid)
#   KVCR_CASES    cases per run (default TC-01..TC-06)
#   KVCR_TIMEOUT  harness timeout per run in seconds (default 3600)
#   KVCR_IMAGE    optional existing local image to reuse instead of building
#   KVCR_EXTRA    optional extra run_phase1.sh args for every config, e.g.
#                 "--harness-arg=--batch-invariant"
#   KVCR_TOPOLOGY single (default) or sidecar: kvcr_service per worker in its
#                 own container, sharing /run/kvcr and /dev/shm/kvcr with the
#                 vLLM container and the host network namespace (Dynamo PR 14695
#                 pod layout); PID namespaces stay separate.
set -euo pipefail

: "${KVCR_BUNDLE:?}" "${KVCR_REF:?}"
MATRIX=${KVCR_MATRIX:-"dense hybrid"}
# TC-05 runs before TC-04 on purpose: TC-04 replaces worker 0 at the same endpoint
# and the first peer transfer afterwards fails (KVCR stale-peer bug), which would
# pollute the remote-hit latency sample. Numbering is unchanged.
CASES=${KVCR_CASES:-TC-01:TC-02:TC-03:TC-05:TC-04:TC-06}
CASES=${CASES//:/,}   # sbatch --export splits on commas, so submit.sh sends colons
TIMEOUT=${KVCR_TIMEOUT:-3600}
RUNS=/home/doxu/kvcr-runs/$SLURM_JOB_ID
W=/tmp/doxu-kvcr-$SLURM_JOB_ID
mkdir -p "$RUNS" "$W"
log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

log "node=$(hostname) kernel=$(uname -r) job=$SLURM_JOB_ID ref=$KVCR_REF"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv | tee "$RUNS/gpus.csv"

# --- preflight: docker with GPU support on this node --------------------------
if ! command -v docker >/dev/null; then
  log "FATAL: docker is not installed on $(hostname); resubmit with --nodelist=<docker node>"
  exit 3
fi
docker info --format 'docker {{.ServerVersion}} driver={{.Driver}} cgroup={{.CgroupDriver}}' | tee "$RUNS/docker.txt"
# Containers run under dockerd, not the Slurm job cgroup: a cancelled earlier job
# can leave a harness container holding the GPUs and ports. Sweep ours (node is
# exclusive) and make sure this job's own container dies with the job.
sweep() { docker ps -aq --filter label=io.ai-dynamo.kvcr.phase1-run | xargs -r docker rm -f >/dev/null 2>&1 || true; }
leftover=$(docker ps -q --filter label=io.ai-dynamo.kvcr.phase1-run | wc -l)
[[ "$leftover" -eq 0 ]] || log "removing $leftover leftover phase1 container(s)"
sweep
trap 'log "job ending (signal/exit); removing phase1 containers"; sweep' EXIT TERM INT
df -h /tmp /var/lib/docker /dev/shm | tee -a "$RUNS/docker.txt"
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}} {{.CreatedSince}}' | grep -i kvcr | tee "$RUNS/images-before.txt" || true

# --- source -------------------------------------------------------------------
git clone -q "$KVCR_BUNDLE" "$W/repo"
git -C "$W/repo" checkout -q --detach "$KVCR_REF"
test "$(git -C "$W/repo" rev-parse HEAD)" = "$KVCR_REF"
cd "$W/repo"

# --- image: reuse a local build of this exact KVCR SHA, else build -------------
IMAGE_REPO=local/kvcr-phase1
export KVCR_E2E_IMAGE=$IMAGE_REPO KVCR_PUSH=0 E2E_BUILD_OUTPUT="$W/artifacts/build"
mkdir -p "$E2E_BUILD_OUTPUT"
if [[ -n ${KVCR_IMAGE:-} ]] && docker image inspect "$KVCR_IMAGE" >/dev/null 2>&1; then
  log "reusing requested image $KVCR_IMAGE (harness code is bind-mounted, so scripts-only commits need no rebuild)"
  IMAGE_ARG=(--image "$KVCR_IMAGE")
elif docker image inspect "$IMAGE_REPO:$KVCR_REF" >/dev/null 2>&1; then
  log "reusing image $IMAGE_REPO:$KVCR_REF"
  IMAGE_ARG=(--image "$IMAGE_REPO:$KVCR_REF")
else
  log "building image (~6 min)"
  bash .github/scripts/build_vllm_e2e.sh 2>&1 | tail -20
  IMAGE_ARG=()
fi
cp -r "$E2E_BUILD_OUTPUT" "$RUNS/build" 2>/dev/null || true

# --- GPUs handed to the container (UUIDs of the two allocated devices) --------
# Only the GPUs Slurm allocated: some nodes lack device-cgroup isolation, so
# nvidia-smi also lists another tenant's GPU (job 2333497 ran a worker on one).
ALLOC=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}}
if [[ -n "$ALLOC" ]]; then
  mapfile -t GPU_UUIDS < <(for i in ${ALLOC//,/ }; do nvidia-smi -i "$i" --query-gpu=uuid --format=csv,noheader; done | head -2)
else
  mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -2)
fi
log "allocated GPUs (${ALLOC:-all visible}): ${GPU_UUIDS[*]}"
case "${#GPU_UUIDS[@]}" in
  2) GPUS="${GPU_UUIDS[0]},${GPU_UUIDS[1]}"; GPU_ARGS=() ;;
  1) # single allocated GPU: both workers share it at reduced utilization
     GPUS="${GPU_UUIDS[0]}"; GPU_ARGS=("--harness-arg=--gpu-memory-utilization=${KVCR_GPU_UTIL:-0.35}")
     log "one GPU allocated: both workers share $GPUS at utilization ${KVCR_GPU_UTIL:-0.35}" ;;
  *) log "FATAL: need one or two GPUs, saw ${GPU_UUIDS[*]}"; exit 4 ;;
esac
CPUS=$(nproc); MEM_G=$(( $(free -g | awk '/^Mem:/{print $2}') * 8 / 10 ))
mkdir -p "$W/hf-cache"

TOPOLOGY=${KVCR_TOPOLOGY:-single}
IMAGE_REF=${KVCR_IMAGE:-}
[[ -n "$IMAGE_REF" ]] || IMAGE_REF=$(python3 -c "import json;d=json.load(open('$E2E_BUILD_OUTPUT/image-identity.json'));print(d.get('local_image_id') or d['image_uri'])")
SIDECAR_ARGS=()

start_sidecars() {
  local name=$1 digest=$2
  RUNKVCR="$W/run-kvcr-$name"; SHM="/dev/shm/kvcr-phase1-$SLURM_JOB_ID-$name"
  mkdir -p "$RUNKVCR" "$SHM/0" "$SHM/1"; chmod 1777 "$RUNKVCR" "$SHM"
  for i in 0 1; do
    docker run -d --name "kvcr-svc-$name-$i" \
      --label "io.ai-dynamo.kvcr.phase1-run=sidecar-$SLURM_JOB_ID" \
      --network host --user root --ulimit memlock=-1 \
      -v "$RUNKVCR:/run/kvcr" -v "$SHM:/dev/shm/kvcr" \
      --entrypoint python3 "$IMAGE_REF" -m kvcr.kvcr_service \
        --socket-path "/run/kvcr/memory-$i.sock" --pool-dir "/dev/shm/kvcr/$i" \
        --guard-count 1 --pool-sizes-gb 2 --compatibility-digest "$digest" --log-level DEBUG >/dev/null
    docker logs -f "kvcr-svc-$name-$i" > "$RUNKVCR/kvcr-service-$i.log" 2>&1 &
  done
  for i in 0 1; do
    for _ in $(seq 1 60); do [[ -S "$RUNKVCR/memory-$i.sock" ]] && break; sleep 1; done
    [[ -S "$RUNKVCR/memory-$i.sock" ]] || { log "FATAL: sidecar $i socket never appeared"; docker logs "kvcr-svc-$name-$i" | tail -20; exit 6; }
  done
  log "sidecars up: $(docker ps --format '{{.Names}} {{.Status}}' | grep kvcr-svc- | tr '\n' ';')"
  SIDECAR_ARGS=(--network host
    # "--docker-arg=-v" form: argparse would otherwise read a bare "-v" as an option
    --docker-arg=-v "--docker-arg=$RUNKVCR:/run/kvcr"
    --docker-arg=-v "--docker-arg=$SHM:/dev/shm/kvcr"
    --harness-arg=--external-services=/run/kvcr
    --harness-arg="--compatibility-digest=$digest")
}

stop_sidecars() {
  local name=$1
  docker ps -aq --filter "name=kvcr-svc-$name-" | xargs -r docker rm -f >/dev/null 2>&1 || true
  mkdir -p "$RUNS/phase1-$name"; cp "$RUNKVCR"/kvcr-service-*.log "$RUNS/phase1-$name/" 2>/dev/null || true
  rm -rf "$SHM" 2>/dev/null || true
}

run_config() {
  local name=$1; shift
  local out="$W/artifacts/phase1-$name"
  SIDECAR_ARGS=()
  if [[ "$TOPOLOGY" == sidecar ]]; then start_sidecars "$name" "kvcr-phase1-$SLURM_JOB_ID-$name"; fi
  log "=== run $name ($TOPOLOGY): cases=$CASES ==="
  set +e
  bash .github/scripts/run_phase1.sh \
    --gpus "$GPUS" "${IMAGE_ARG[@]}" \
    --build-output "$E2E_BUILD_OUTPUT" \
    --output-dir "$out" \
    --cpus "$CPUS" --memory "${MEM_G}g" \
    --hf-cache "$W/hf-cache" --timeout "$TIMEOUT" \
    --cases "$CASES" "${GPU_ARGS[@]}" "${SIDECAR_ARGS[@]}" "$@" 2>&1 | tail -5
  local rc=${PIPESTATUS[0]}
  set -e
  log "=== run $name exit=$rc ==="
  cp -r "$out" "$RUNS/phase1-$name"
  if [[ "$TOPOLOGY" == sidecar ]]; then stop_sidecars "$name"; fi
  python3 - "$out/run/result.json" <<'PY' | tee "$RUNS/phase1-$name.summary.txt" || true
import json, sys
r = json.load(open(sys.argv[1]))
print("status", r.get("status"), "model", r.get("model"), "error", r.get("error"))
for c in r.get("cases", []):
    bad = [k["name"] + ": " + k["detail"] for k in c["checks"] if not k["passed"]]
    print(f"  {c['case']:5} {c['status']:7} checks={len(c['checks'])} failed={len(bad)}")
    for b in bad[:6]:
        print("      -", b[:200])
    m = c.get("measurements", {})
    for arm in [k for k in m if k.startswith("arm_")]:
        s = m[arm]
        if isinstance(s, dict) and "ttft_seconds" in s:
            t, p = s["ttft_seconds"], s.get("tpot_seconds", {})
            print(f"      {arm:38} ttft p50={t.get('median')} p95={t.get('p95')}  tpot p50={p.get('median')}  n={t.get('n')}")
    if "timeline_seconds" in m:
        print("      timeline", m["timeline_seconds"])
    if "derived" in m:
        print("      derived", {k: v for k, v in m["derived"].items() if k != "note"})
PY
}

for cfg in $MATRIX; do
  # shellcheck disable=SC2086
  case $cfg in
    dense)  run_config dense  --harness-arg=--model=Qwen/Qwen3-0.6B ${KVCR_EXTRA:-} ;;
    # Qwen3.5-4B geometry (measured 2026-10-08/19): attention block raised to
    # 576 tokens, 3 SSM groups, 18 MiB KVCR pages; the flags make every
    # delivered-pages and bytes check exact instead of a lower bound.
    hybrid) run_config hybrid --harness-arg=--model=Qwen/Qwen3.5-4B \
                              --harness-arg=--worker-arg=--mamba-cache-mode=align \
                              --harness-arg=--keep-hybrid-kv-cache-manager \
                              --harness-arg=--no-enforce-eager \
                              --harness-arg=--hit-block-tokens --harness-arg=576 \
                              --harness-arg=--ssm-groups --harness-arg=3 \
                              --harness-arg=--page-bytes --harness-arg=18874368 \
                              --harness-arg=--tc02-shared-sections --harness-arg=140 \
                              --harness-arg=--tc02-sections --harness-arg=200 \
                              ${KVCR_EXTRA:-} ;;
    *) log "unknown config $cfg"; exit 5 ;;
  esac
done

docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -i kvcr | tee "$RUNS/images-after.txt" || true
tar -czf "$RUNS.tgz" -C "$RUNS/.." "$SLURM_JOB_ID"
log "evidence: $RUNS ($RUNS.tgz)"
