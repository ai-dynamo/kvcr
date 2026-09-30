#!/usr/bin/env python3
"""Render a Kubernetes Pod that runs KVCR test plan Phase 1 in the sidecar layout.

One pod, three containers on the shared pod network namespace, separate PID
namespaces (Dynamo PR 14695 layout):
  kvcr-svc-0 / kvcr-svc-1   python3 -m kvcr.kvcr_service, own the DRAM pools
  main                      vLLM workers + Dynamo frontend + the Phase 1 harness
Shared volumes: /run/kvcr (sockets, service logs) and /dev/shm/kvcr (pool files,
memory-backed). Tests and evidence live on the RWX PVC.

  scripts/k8s/phase1-pod.py --image ghcr.io/luckyq/kvcr-phase1:<sha> \
      --tests-sha df08797 [--cases ...] [--model Qwen/Qwen3-0.6B] \
      [--worker-arg ...] [--no-batch-invariant] > pod.yaml
"""
import argparse
import json
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--tests-sha", required=True, help="tests tarball already extracted to PVC:/doxu-kvcr/tests")
p.add_argument("--name", default=None)
p.add_argument("--namespace", default="doxu-dynamo")
p.add_argument("--pvc", default="shared-model-cache")
p.add_argument("--pull-secret", default="ghcr-luckyq")
p.add_argument("--cases", default="TC-01,TC-02,TC-03,TC-05,TC-04,TC-06")
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--timeout", type=int, default=3600)
p.add_argument("--pool-gb", type=int, default=2)
p.add_argument("--no-batch-invariant", action="store_true")
p.add_argument("--harness-arg", action="append", default=[])
p.add_argument("--gpu-type", default="H100", help="node label nvidia.com/gpu.product substring, '' to skip")
args = p.parse_args()

run_id = args.name or f"kvcr-phase1-{time.strftime('%m%d-%H%M')}"
digest = f"{run_id}-{args.tests_sha}"
tests = f"/cache/doxu-kvcr/tests/tests/e2e"
evidence = f"/cache/doxu-kvcr/evidence/{run_id}"

harness = [
    "python3", f"{tests}/run_phase1.py",
    "--output-dir", f"{evidence}/run",
    "--mode", "guard",
    "--cases", args.cases,
    "--timeout", str(args.timeout),
    "--gpus", "0,1",
    "--model", args.model,
    "--pool-gb", str(args.pool_gb),
    "--external-services", "/run/kvcr",
    "--compatibility-digest", digest,
    *([] if args.no_batch_invariant else ["--batch-invariant"]),
    *args.harness_arg,
]


def sidecar(i: int) -> dict:
    cmd = (
        f"mkdir -p /dev/shm/kvcr/{i} && exec python3 -m kvcr.kvcr_service "
        f"--socket-path /run/kvcr/memory-{i}.sock --pool-dir /dev/shm/kvcr/{i} "
        f"--guard-count 1 --pool-sizes-gb {args.pool_gb} --compatibility-digest {digest} "
        f"--log-level DEBUG > /run/kvcr/kvcr-service-{i}.log 2>&1"
    )
    return {
        "name": f"kvcr-svc-{i}",
        "image": args.image,
        "imagePullPolicy": "IfNotPresent",
        # native sidecar: starts before main, is stopped when main exits
        "restartPolicy": "Always",
        "command": ["sh", "-c", cmd],
        "securityContext": {"runAsUser": 0},
        "resources": {
            "requests": {"cpu": "1", "memory": "2Gi"},
            "limits": {"cpu": "4", "memory": f"{args.pool_gb * 2 + 4}Gi"},
        },
        "volumeMounts": [
            {"name": "kvcr-run", "mountPath": "/run/kvcr"},
            {"name": "kvcr-shm", "mountPath": "/dev/shm/kvcr"},
        ],
    }


pod = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {
        "name": run_id,
        "namespace": args.namespace,
        "labels": {"app": "kvcr-phase1", "owner": "doxu", "tests-sha": args.tests_sha},
        "annotations": {"kvcr.phase1/harness": json.dumps(harness)},
    },
    "spec": {
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": args.pull_secret}],
        "terminationGracePeriodSeconds": 30,
        **({"nodeSelector": {"nvidia.com/gpu.product": f"NVIDIA-{args.gpu_type}-80GB-HBM3"}} if args.gpu_type else {}),
        "initContainers": [sidecar(0), sidecar(1)],
        "containers": [
            {
                "name": "main",
                "image": args.image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["sh", "-c",
                            f"mkdir -p {evidence} && ulimit -l unlimited 2>/dev/null; ulimit -s 65536 2>/dev/null; "
                            "cd /tmp && exec \"$@\"", "--", *harness],
                "workingDir": "/tmp",
                "securityContext": {"runAsUser": 0},
                "env": [
                    {"name": "HF_HOME", "value": "/cache/doxu-kvcr/hf"},
                    {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                ],
                "resources": {
                    "requests": {"cpu": "12", "memory": "64Gi", "nvidia.com/gpu": "2"},
                    "limits": {"cpu": "24", "memory": "120Gi", "nvidia.com/gpu": "2"},
                },
                "volumeMounts": [
                    {"name": "cache", "mountPath": "/cache"},
                    {"name": "dshm", "mountPath": "/dev/shm"},
                    {"name": "kvcr-run", "mountPath": "/run/kvcr"},
                    {"name": "kvcr-shm", "mountPath": "/dev/shm/kvcr"},
                ],
            }
        ],
        "volumes": [
            {"name": "cache", "persistentVolumeClaim": {"claimName": args.pvc}},
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
            {"name": "kvcr-run", "emptyDir": {}},
            {"name": "kvcr-shm", "emptyDir": {"medium": "Memory", "sizeLimit": f"{args.pool_gb * 2 + 2}Gi"}},
        ],
    },
}
json.dump(pod, sys.stdout, indent=1)
print()
print(f"# evidence: PVC:{evidence}  digest: {digest}", file=sys.stderr)
