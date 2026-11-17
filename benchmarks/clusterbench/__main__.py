# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""clusterbench command line.

local        run a whole cell on this machine
ab           interleaved local A/B (or A/A) between KVCR source trees
coordinator  run the coordinator of a multi-node cell
worker       run one worker (started by a launcher)
report       print one run's summary, or compare runs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="scenario TOML")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="A.B=V",
        help="override a config field, e.g. phases.measure_s=30",
    )


def _local(args) -> int:
    from .config import load_config
    from .launch_local import run_local
    from .report import format_summary

    config = load_config(args.config, args.set)
    out = args.out or f"runs/{config.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    summary = run_local(
        config,
        out,
        nodes=args.nodes,
        per_node=args.per_node,
        kvcr_src=args.kvcr_src,
        arm=args.arm,
        cores_per_worker=args.cores_per_worker,
    )
    print(format_summary(summary))
    print(f"\nrun directory: {out}")
    return 0 if summary["valid"] or args.allow_invalid else 2


def _ab(args) -> int:
    from .config import load_config
    from .launch_local import run_local
    from .report import compare

    config = load_config(args.config, args.set)
    arms = [item.partition("=")[::2] for item in args.arm]
    if len(arms) < 2 or any(not src for _, src in arms):
        raise SystemExit("need at least two --arm NAME=KVCR_SOURCE_TREE")
    root = Path(args.out or f"runs/ab-{time.strftime('%Y%m%d-%H%M%S')}")
    results: dict[str, list[dict]] = {name: [] for name, _ in arms}
    for rep in range(args.reps):
        # Rotate which arm goes first so drift does not favour one arm.
        order = arms[rep % len(arms) :] + arms[: rep % len(arms)]
        for name, src in order:
            out = root / f"{name}-{rep}"
            print(f"[ab] rep {rep} arm {name}", flush=True)
            results[name].append(
                run_local(
                    config,
                    out,
                    nodes=args.nodes,
                    per_node=args.per_node,
                    kvcr_src=src,
                    arm=name,
                    cores_per_worker=args.cores_per_worker,
                )
            )
    print(compare(results))
    print(f"\nrun directories under: {root}")
    return 0


def _coordinator(args) -> int:
    from .config import load_config
    from .coordinator import Coordinator
    from .report import format_summary

    config = load_config(args.config, args.set)
    coordinator = Coordinator(config, args.out, arm=args.arm)
    try:
        coordinator.bind(args.bind)
        summary = coordinator.run(
            expect=args.expect, expect_sha=args.expect_sha, per_node=args.per_node
        )
    finally:
        coordinator.close()
    print(format_summary(summary))
    return 0 if summary["valid"] or args.allow_invalid else 2


def _worker(args) -> int:
    from .worker import run_worker

    return run_worker(args)


def _report(args) -> int:
    from .report import compare, format_detail, format_summary, load

    if args.arm:
        arms = {}
        for item in args.arm:
            name, _, dirs = item.partition("=")
            arms[name] = [load(d) for d in dirs.split(",") if d]
        print(compare(arms))
        return 0
    for run in args.runs:
        summary = load(run)
        print(format_summary(summary))
        print()
        if args.detail:
            print(format_detail(summary))
            print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clusterbench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def topology(p: argparse.ArgumentParser) -> None:
        p.add_argument("--nodes", type=int, default=2, help="emulated nodes")
        p.add_argument("--per-node", type=int, default=2, help="workers per node")
        p.add_argument(
            "--cores-per-worker",
            type=int,
            default=0,
            help="pin each worker to this many physical cores (>=3; >=4 for sglang)",
        )
        p.add_argument("--out", help="run directory")

    local = sub.add_parser("local", help="run a whole cell on this machine")
    _add_config(local)
    topology(local)
    local.add_argument(
        "--kvcr-src", help="KVCR source tree to test (default: the installed kvcr)"
    )
    local.add_argument("--arm", default="")
    local.add_argument("--allow-invalid", action="store_true")
    local.set_defaults(func=_local)

    ab = sub.add_parser("ab", help="interleaved local A/B between KVCR trees")
    _add_config(ab)
    topology(ab)
    ab.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=TREE",
        help="repeat once per arm",
    )
    ab.add_argument("--reps", type=int, default=3)
    ab.set_defaults(func=_ab)

    coord = sub.add_parser("coordinator", help="coordinator of a multi-node cell")
    _add_config(coord)
    coord.add_argument("--bind", required=True, help="e.g. tcp://0.0.0.0:17700")
    coord.add_argument("--expect", type=int, required=True, help="worker count")
    coord.add_argument("--per-node", type=int, default=0, help="workers per node")
    coord.add_argument("--expect-sha", default="")
    coord.add_argument("--out", required=True)
    coord.add_argument("--arm", default="")
    coord.add_argument("--allow-invalid", action="store_true")
    coord.set_defaults(func=_coordinator)

    worker = sub.add_parser("worker", help="one worker (started by a launcher)")
    worker.add_argument("--coordinator", required=True)
    worker.add_argument("--node", required=True)
    worker.add_argument("--local-id", type=int, required=True)
    worker.add_argument("--bind-host", default="127.0.0.1")
    worker.add_argument("--advertise-host", default="")
    worker.add_argument("--control-port", type=int, required=True)
    worker.add_argument("--nixl-port", type=int, required=True)
    worker.add_argument(
        "--cpus",
        default="",
        help='CPU list (e.g. 4-11,76-83), or "auto" with --per-node',
    )
    worker.add_argument("--per-node", type=int, default=1, help="workers on this node")
    worker.add_argument("--reserve-cores", type=int, default=2, help="per socket")
    worker.add_argument(
        "--net-devices", default="", help="UCX_NET_DEVICES per socket, comma separated"
    )
    worker.add_argument("--arm", default="")
    worker.add_argument("--configure-timeout", type=int, default=600)
    worker.set_defaults(func=_worker)

    report = sub.add_parser("report", help="summarise or compare runs")
    report.add_argument("runs", nargs="*")
    report.add_argument(
        "--detail", action="store_true", help="all histograms and counters"
    )
    report.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=DIR[,DIR...]",
        help="compare arms; the first is the baseline",
    )
    report.set_defaults(func=_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
