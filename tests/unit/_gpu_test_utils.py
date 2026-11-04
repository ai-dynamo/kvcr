# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from _kvcr_test_utils import FakeNixlAgent

DOMAIN = "test-host:cuda-ipc:layout-v1"
TLS = "cuda_ipc,posix,self"


class GPUAgent(FakeNixlAgent):
    def __init__(self):
        super().__init__()
        self.prep_types = []
        self.plugin_params = {"ucx_error_handling_mode": "peer", "num_workers": "1"}
        self.backend_params = {}
        self.backend_calls = []
        self.backends = {}

    def get_plugin_params(self, backend):
        assert backend == "UCX"
        self.backend_calls.append(("defaults", backend))
        return self.plugin_params

    def create_backend(self, backend, params):
        assert backend == "UCX"
        self.backend_calls.append(("create", backend, dict(params)))
        self.backend_params = dict(params)
        self.backends[backend] = object()

    def get_backend_params(self, backend):
        assert backend == "UCX"
        self.backend_calls.append(("readback", backend))
        return self.backend_params

    def get_backend_mem_types(self, backend):
        assert backend == "UCX"
        self.backend_calls.append(("memory_types", backend))
        return ["DRAM_SEG", "VRAM_SEG"]

    def prep_xfer_dlist(self, agent_name, descs, *, mem_type, backends):
        self.prep_types.append(mem_type)
        return super().prep_xfer_dlist(
            agent_name, descs, mem_type=mem_type, backends=backends
        )

    def transfer(self, handle):
        # Opaque CUDA addresses must never be dereferenced by the fake backend.
        self.transfers.append(handle)
        return "PROC"
