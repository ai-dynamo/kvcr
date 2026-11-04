# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from _gpu_test_utils import DOMAIN, GPUAgent, GPUPinning, regions
from _kvcr_test_utils import FakeBytesControl, _new_kvcr

from kvcr.config import KVCRConfig, RemoteFWGPUOptions
from kvcr.runtime.progress import _KVCRProgress


def test_mnnvl_constructor_rejects_legacy_global_policy_before_native_work(
    monkeypatch,
):
    monkeypatch.setenv("NIXL_UCX_CUDA_IPC_ONLY", "1")
    agent, pinning = GPUAgent(), GPUPinning()
    options = RemoteFWGPUOptions(
        compatibility_domain=DOMAIN,
        gpu_transport="mnnvl",
        fabric_domain="mnnvl:" + "1" * 32 + ":0",
        ucx_tls=("cuda_ipc", "tcp", "self"),
    )
    with pytest.raises(ValueError, match="global CUDA IPC-only policy"):
        _new_kvcr(
            agent,
            pinning,
            FakeBytesControl(),
            KVCRConfig(
                nixl_agent_name="target", pool_layouts=[("main", 16), ("index", 8)]
            ),
            framework_regions=regions("target"),
            remote_gpu_options=options,
        )
    assert agent.backend_calls == agent.registrations == agent.xfers == []
    assert pinning.searches == []


@pytest.mark.parametrize(
    "transport,tls",
    [
        ("rdma", None),
        ("mnnvl", None),
        ("mnnvl", ("cuda_ipc", "posix", "self")),
    ],
)
def test_gpu_progress_rejects_unbound_transport_policy(transport, tls):
    with pytest.raises(ValueError, match="GPU transport|MNNVL requires"):
        _KVCRProgress(
            Mock(),
            Mock(),
            list,
            Mock(),
            required_gpu_transport=transport,
            required_gpu_ucx_tls=tls,
        )
