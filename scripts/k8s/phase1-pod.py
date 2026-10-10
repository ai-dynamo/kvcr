#!/usr/bin/env python3
"""Render Kubernetes objects that run KVCR test plan Phase 1 in the sidecar layout.

Single node (default): one pod, 2 GPUs.
  kvcr-svc-0 / kvcr-svc-1   python3 -m kvcr.kvcr_service, own the DRAM pools (native sidecars)
  main                      vLLM workers + Dynamo frontend + the Phase 1 harness

Two nodes (--two-node): two pods on different nodes, 1 GPU each, plus headless
Services for DNS.
  <run>-b   kvcr-svc-1 sidecar + node agent (tests/e2e/phase1/node_agent.py) that
            spawns worker 1 on the harness's behalf
  <run>-a   etcd sidecar (Dynamo discovery; the file store's inotify watch is
            node-local) + kvcr-svc-0 sidecar + main (frontend, state agent,
            worker 0, harness with --remote-worker 1=http://<run>-b:9800)

Shared inside a pod: /run/kvcr (sockets, service logs) and /dev/shm/kvcr (pool
files, memory-backed). Tests, evidence and the HF cache live on the RWX PVC.

  scripts/k8s/phase1-pod.py --image ghcr.io/luckyq/kvcr-phase1:<sha> --tests-sha <sha> \
      [--two-node] [--cases ...] [--model Qwen/Qwen3-0.6B] [--harness-arg ...] > objects.json
"""
from __future__ import annotations

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
p.add_argument("--two-node", action="store_true")
p.add_argument("--topology", choices=("agg", "disagg"), default="agg",
               help="disagg = 2P+1D on three GPUs in one pod (workers 0/1 prefill with KVCR sidecars, worker 2 decode)")
p.add_argument("--etcd-image", default="quay.io/coreos/etcd:v3.5.14")
p.add_argument("--agent-port", type=int, default=9800)
p.add_argument("--gpu-type", default="H100", help="node label nvidia.com/gpu.product substring, '' to skip")
p.add_argument("--overlay", action="append", default=[], metavar="PVC_PATH:IMAGE_PATH",
               help="copy a file from the PVC over a file in the image before the engine starts "
                    "(patch one vLLM/Dynamo module without rebuilding); applied in every GPU container")
p.add_argument("--extra-env", action="append", default=[], metavar="KEY=VALUE",
               help="environment for every container, e.g. PYTHONPATH=/cache/doxu-kvcr/fix51/src "
                    "to test a patched KVCR without rebuilding the image; repeatable")
args = p.parse_args()
overlay_cmd = ""
for item in args.overlay:
    src, sep, dst = item.partition(":")
    if not sep or not src.startswith("/") or not dst.startswith("/"):
        p.error(f"--overlay expects /pvc/path:/image/path, got {item!r}")
    overlay_cmd += f"cp {src} {dst} && "
extra_env = []
for item in args.extra_env:
    key, sep, value = item.partition("=")
    if not sep or not key:
        p.error(f"--extra-env expects KEY=VALUE, got {item!r}")
    extra_env.append({"name": key, "value": value})

run_id = args.name or f"kvcr-phase1-{time.strftime('%m%d-%H%M')}"
ns = args.namespace
digest = f"{run_id}-{args.tests_sha}"
tests = "/cache/doxu-kvcr/tests/tests/e2e"
evidence = f"/cache/doxu-kvcr/evidence/{run_id}"
stack_dir = f"{evidence}/run/stack"  # Stack.evidence: where a remote sidecar must log
labels = {"app": "kvcr-phase1", "owner": "doxu", "run": run_id, "tests-sha": args.tests_sha}
node_selector = (
    {"nvidia.com/gpu.product": f"NVIDIA-{args.gpu_type}-80GB-HBM3"} if args.gpu_type else {}
)
if args.two_node and args.topology == "disagg":
    p.error("--topology disagg is single-node only")
gpus_per_pod = 1 if args.two_node else (3 if args.topology == "disagg" else 2)

harness = [
    "python3", f"{tests}/run_phase1.py",
    "--output-dir", f"{evidence}/run",
    "--mode", "guard",
    "--cases", args.cases,
    "--timeout", str(args.timeout),
    "--gpus", "0,0" if args.two_node else ("0,1,2" if args.topology == "disagg" else "0,1"),
    "--topology", args.topology,
    "--model", args.model,
    "--pool-gb", str(args.pool_gb),
    "--external-services", "/run/kvcr",
    "--compatibility-digest", digest,
    *([] if args.no_batch_invariant else ["--batch-invariant"]),
]
if args.two_node:
    harness += [
        "--remote-worker", f"1=http://{run_id}-b.{ns}.svc.cluster.local:{args.agent_port}",
        "--etcd", f"http://{run_id}-a.{ns}.svc.cluster.local:2379",
    ]
harness += args.harness_arg

common_mounts = [
    {"name": "cache", "mountPath": "/cache"},
    {"name": "dshm", "mountPath": "/dev/shm"},
    {"name": "kvcr-run", "mountPath": "/run/kvcr"},
    {"name": "kvcr-shm", "mountPath": "/dev/shm/kvcr"},
]
volumes = [
    {"name": "cache", "persistentVolumeClaim": {"claimName": args.pvc}},
    {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
    {"name": "kvcr-run", "emptyDir": {}},
    {"name": "kvcr-shm", "emptyDir": {"medium": "Memory", "sizeLimit": f"{args.pool_gb * 2 + 2}Gi"}},
]


def sidecar(i: int, log_path: str) -> dict:
    cmd = (
        f"mkdir -p /dev/shm/kvcr/{i} $(dirname {log_path}) && exec python3 -m kvcr.kvcr_service "
        f"--socket-path /run/kvcr/memory-{i}.sock --pool-dir /dev/shm/kvcr/{i} "
        f"--guard-count 1 --pool-sizes-gb {args.pool_gb} --compatibility-digest {digest} "
        f"--log-level DEBUG > {log_path} 2>&1"
    )
    return {
        "name": f"kvcr-svc-{i}",
        "image": args.image,
        "imagePullPolicy": "IfNotPresent",
        "restartPolicy": "Always",  # native sidecar: up before main, stopped after it
        "command": ["sh", "-c", cmd],
        "securityContext": {"runAsUser": 0},
        **({"env": extra_env} if extra_env else {}),
        "resources": {
            "requests": {"cpu": "1", "memory": "2Gi"},
            "limits": {"cpu": "4", "memory": f"{args.pool_gb * 2 + 4}Gi"},
        },
        "volumeMounts": [
            {"name": "cache", "mountPath": "/cache"},
            {"name": "kvcr-run", "mountPath": "/run/kvcr"},
            {"name": "kvcr-shm", "mountPath": "/dev/shm/kvcr"},
        ],
    }


def etcd_sidecar() -> dict:
    return {
        "name": "etcd",
        "image": args.etcd_image,
        "imagePullPolicy": "IfNotPresent",
        "restartPolicy": "Always",
        # distroless image, no shell; kubelet expands $(POD_IP) from env
        "command": ["etcd", "--data-dir", "/tmp/etcd",
                    "--listen-client-urls", "http://0.0.0.0:2379",
                    "--advertise-client-urls", "http://$(POD_IP):2379",
                    "--listen-peer-urls", "http://127.0.0.1:2380"],
        "env": [{"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}],
        "resources": {"requests": {"cpu": "500m", "memory": "512Mi"},
                      "limits": {"cpu": "2", "memory": "2Gi"}},
        "readinessProbe": {"tcpSocket": {"port": 2379}, "periodSeconds": 2},
    }


def gpu_main(name: str, command: list, probe: dict | None = None) -> dict:
    return {
        "name": name,
        "image": args.image,
        "imagePullPolicy": "IfNotPresent",
        "command": command,
        "workingDir": "/tmp",
        "securityContext": {"runAsUser": 0},
        "env": [
            {"name": "HF_HOME", "value": "/cache/doxu-kvcr/hf"},
            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
            *extra_env,
        ],
        "resources": {
            "requests": {"cpu": "12", "memory": "48Gi", "nvidia.com/gpu": str(gpus_per_pod)},
            "limits": {"cpu": "24", "memory": "120Gi", "nvidia.com/gpu": str(gpus_per_pod)},
        },
        **({"readinessProbe": probe} if probe else {}),
        "volumeMounts": common_mounts,
    }


def pod(name: str, role: str, init: list, main: dict, anti_affinity: bool) -> dict:
    spec = {
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": args.pull_secret}],
        "terminationGracePeriodSeconds": 30,
        **({"nodeSelector": node_selector} if node_selector else {}),
        "initContainers": init,
        "containers": [main],
        "volumes": volumes,
    }
    if anti_affinity:
        spec["affinity"] = {"podAntiAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": [
            {"labelSelector": {"matchLabels": {"run": run_id}}, "topologyKey": "kubernetes.io/hostname"}
        ]}}
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": name, "namespace": ns, "labels": {**labels, "role": role},
                     "annotations": {"kvcr.phase1/harness": json.dumps(harness)}},
        "spec": spec,
    }


def headless(name: str, role: str, port: int, port_name: str) -> dict:
    return {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": name, "namespace": ns, "labels": {**labels, "role": role}},
        "spec": {"clusterIP": "None", "publishNotReadyAddresses": True,
                 "selector": {"run": run_id, "role": role},
                 "ports": [{"name": port_name, "port": port, "targetPort": port}]},
    }


main_cmd = ["sh", "-c",
            overlay_cmd + f"mkdir -p {evidence} && ulimit -l unlimited 2>/dev/null; ulimit -s 65536 2>/dev/null; "
            "cd /tmp && exec \"$@\"", "--", *harness]

items: list[dict] = []
if not args.two_node:
    items.append(pod(run_id, "single",
                     [sidecar(0, "/run/kvcr/kvcr-service-0.log"), sidecar(1, "/run/kvcr/kvcr-service-1.log")],
                     gpu_main("main", main_cmd), anti_affinity=False))
else:
    agent_cmd = ["sh", "-c",
                 overlay_cmd + f"ulimit -l unlimited 2>/dev/null; ulimit -s 65536 2>/dev/null; cd {tests} && "
                 f"exec python3 -m phase1.node_agent --port {args.agent_port}"]
    items.append(headless(f"{run_id}-b", "b", args.agent_port, "agent"))
    items.append(pod(f"{run_id}-b", "b",
                     [sidecar(1, f"{stack_dir}/kvcr-service-1.log")],
                     gpu_main("node-agent", agent_cmd,
                              probe={"httpGet": {"path": "/info", "port": args.agent_port}, "periodSeconds": 2}),
                     anti_affinity=True))
    items.append(headless(f"{run_id}-a", "a", 2379, "etcd"))
    items.append(pod(f"{run_id}-a", "a",
                     [etcd_sidecar(), sidecar(0, "/run/kvcr/kvcr-service-0.log")],
                     gpu_main("main", main_cmd), anti_affinity=True))

json.dump({"apiVersion": "v1", "kind": "List", "items": items}, sys.stdout, indent=1)
print()
print(f"# run: {run_id}  evidence: PVC:{evidence}  digest: {digest}  "
      f"topology: {'two-node' if args.two_node else 'single-node'}", file=sys.stderr)
