#!/bin/bash
#SBATCH --job-name=doxu-kvcr-phase1
#SBATCH --partition=general
#SBATCH --gres=gpu:H100:2
#SBATCH --constraint="2GPU&x86_64&UB24.04"
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/home/doxu/kvcr-runs/%j.out
#
# KVCR test plan Phase 1 (TC-01..TC-06) on one dlcluster node with 2x H100.
# Exclusive 2-GPU node: shared 8-GPU nodes hand out GPUs with other tenants'
# memory on them and vLLM's 0.8 utilization check then fails at startup.
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
set -euo pipefail

: "${KVCR_BUNDLE:?}" "${KVCR_REF:?}"
MATRIX=${KVCR_MATRIX:-"dense hybrid"}
CASES=${KVCR_CASES:-TC-01:TC-02:TC-03:TC-04:TC-05:TC-06}
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
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -2)
test "${#GPU_UUIDS[@]}" -eq 2 || { log "FATAL: need two GPUs, saw ${GPU_UUIDS[*]}"; exit 4; }
GPUS="${GPU_UUIDS[0]},${GPU_UUIDS[1]}"
CPUS=$(nproc); MEM_G=$(( $(free -g | awk '/^Mem:/{print $2}') * 8 / 10 ))
mkdir -p "$W/hf-cache"

run_config() {
  local name=$1; shift
  local out="$W/artifacts/phase1-$name"
  log "=== run $name: cases=$CASES ==="
  set +e
  bash .github/scripts/run_phase1.sh \
    --gpus "$GPUS" "${IMAGE_ARG[@]}" \
    --build-output "$E2E_BUILD_OUTPUT" \
    --output-dir "$out" \
    --cpus "$CPUS" --memory "${MEM_G}g" \
    --hf-cache "$W/hf-cache" --timeout "$TIMEOUT" \
    --cases "$CASES" "$@" 2>&1 | tail -5
  local rc=${PIPESTATUS[0]}
  set -e
  log "=== run $name exit=$rc ==="
  cp -r "$out" "$RUNS/phase1-$name"
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
    hybrid) run_config hybrid --harness-arg=--model=Qwen/Qwen3.5-0.8B \
                              --harness-arg=--worker-arg=--mamba-cache-mode=align \
                              --harness-arg=--keep-hybrid-kv-cache-manager ${KVCR_EXTRA:-} ;;
    *) log "unknown config $cfg"; exit 5 ;;
  esac
done

docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -i kvcr | tee "$RUNS/images-after.txt" || true
tar -czf "$RUNS.tgz" -C "$RUNS/.." "$SLURM_JOB_ID"
log "evidence: $RUNS ($RUNS.tgz)"
