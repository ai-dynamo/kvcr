# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check installed vLLM/Dynamo rank initialization without GPUs or native I/O.

Run inside the serving image after the ordinary worker-entrypoint import canary.
Only native KVCR construction and the control channel are mocked. The installed
adapter constructor, its parent initializer and Dynamo endpoint resolver run.
This preserves rank-dependent initialization checks after the standalone DP=2
GPU smoke is removed; it does not qualify DP=2 serving or transfers at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def check_rank_initialization() -> dict:
    from dynamo.llm import WorkerType
    from dynamo.vllm.kv_hints import resolve_kv_transfer_hint_sources
    from vllm.v1.kv_offload.tiering.kvcr import manager

    ports = [17771, 17772]
    host = "127.0.0.1"
    names = []
    checked = []

    def construct(rank, configured_ports):
        spec = SimpleNamespace(
            config=SimpleNamespace(
                parallel=SimpleNamespace(data_parallel_rank_local=rank)
            ),
            kv_events_config=SimpleNamespace(
                enable_kv_cache_events=True, self_describing_kv_events=True
            ),
        )
        return manager.KVCRSecondaryTierManager(
            spec,
            memoryview(bytearray(64)),
            "kvcr",
            router_capabilities=["router_hint"],
            control_ports=configured_ports,
            control_advertise_host=host,
        )

    # Call the real constructor. No copied port-selection logic substitutes for
    # what the installed adapter actually passes into its channel constructor.
    for rank, configured_ports, expected_port in (
        (0, ports, 17771),
        (1, ports, 17772),
        (None, [17771], 17771),
    ):
        with (
            patch.object(manager, "ZmqPeerControlChannel") as channel,
            patch.object(manager, "KVCR") as native,
        ):
            construct(rank, configured_ports)
            channel.assert_called_once_with("0.0.0.0", expected_port, host)
            native.assert_called_once()
            config, bindings, backends, guard = native.call_args.args
            assert bindings.framework_control is channel.return_value
            assert guard is None  # The removed smoke was native, never Guard.
            assert backends.local_dram is None
            assert config.nixl_agent_name.startswith("KVCR-")
            names.append(config.nixl_agent_name)
            checked.append({"local_rank": rank, "control_port": expected_port})
    assert len(set(names)) == len(names), "rank instances share a transport identity"

    # Invalid configuration must fail before any control/native instance starts.
    invalid = (
        (-1, ports),
        (2, ports),
        (None, ports),
        (None, None),
        (0, []),
        (0, [0]),
        (0, [65536]),
    )
    for rank, configured_ports in invalid:
        with (
            patch.object(manager, "ZmqPeerControlChannel") as channel,
            patch.object(manager, "KVCR") as native,
        ):
            try:
                construct(rank, configured_ports)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"accepted invalid local rank/ports: {rank}, {configured_ports}"
                )
            channel.assert_not_called()
            native.assert_not_called()

    tier = {
        "type": "kvcr",
        "router_capabilities": ["router_hint"],
        "control_ports": ports,
        "control_advertise_host": host,
    }
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="OffloadingConnector",
            kv_connector_extra_config={"secondary_tiers": [tier]},
        )
    )
    published = {}
    for start, expected in (
        (0, {0: "tcp://127.0.0.1:17771", 1: "tcp://127.0.0.1:17772"}),
        (4, {4: "tcp://127.0.0.1:17771", 5: "tcp://127.0.0.1:17772"}),
    ):
        sources = resolve_kv_transfer_hint_sources(
            engine_args, WorkerType.Aggregated, dp_range=(start, 2)
        )
        assert sources is not None
        endpoints = {
            rank: source.source_control_endpoint for rank, source in sources.items()
        }
        assert endpoints == expected, (endpoints, expected)
        assert all(source.worker_type == "aggregated" for source in sources.values())
        published[str(start)] = endpoints

    return {
        "revision": "installed-rank-initialization-v1",
        "status": "PASS",
        "adapter_sha256": hashlib.sha256(
            Path(manager.__file__).read_bytes()
        ).hexdigest(),
        "local_rank_cases": checked,
        "invalid_configurations_rejected": len(invalid),
        "published_source_endpoints": published,
        "distinct_transport_identities": len(names),
        "dp2_runtime_qualified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    result = check_rank_initialization()
    if args.metadata is not None:
        metadata = json.loads(args.metadata.read_text())
        metadata["vllm_rank_initialization"] = result
        args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
