# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File-only CUDA trace extraction. Never promotes an empty trace to proof."""

import argparse
import hashlib
import json
from pathlib import Path


def inspect(trace, peer):
    raw = trace.read_bytes()
    proof = json.loads(peer.read_text())
    assert proof["profile"]["requested"] is True
    assert proof["profile"]["status"] == "TRACE_CAPTURED_UNREVIEWED"
    assert proof["profile"]["trace_sha256"] == hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    events = data["traceEvents"]
    assert isinstance(events, list)
    copies = [
        e
        for e in events
        if "memcpy" in str(e.get("name", "")).lower()
        or "memcpy" in str(e.get("cat", "")).lower()
    ]
    gpu_events = [
        e
        for e in events
        if any(
            word in str(e.get("cat", "")).lower()
            for word in ("gpu_memcpy", "gpu_memset", "kernel")
        )
    ]
    return {
        "status": "UNQUALIFIED_TRACE_EVIDENCE_REQUIRES_INDEPENDENT_ROUTE_REVIEW",
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "peer_result_sha256": hashlib.sha256(peer.read_bytes()).hexdigest(),
        "role": proof["role"],
        "physical_device": proof["physical_device"],
        "cuda_activity_event_count": len(gpu_events),
        "all_memcpy_api_and_activity_events": copies,
        "physical_no_host_staging_qualified": False,
        "empty_or_unpopulated_cuda_trace": not gpu_events,
        "required_followup": (
            "Join actual native memcpy kinds/bytes/devices/lane and complete "
            "coverage to source/target registrations; reject unsupported, empty, "
            "ambiguous or host-payload staging evidence. A target with no "
            "initiated copies does not substitute for source payload evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--peer-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.write_text(
        json.dumps(inspect(args.trace, args.peer_result), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
