# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise native NIXL multi-pool G3 transfers without downloading a model."""

import argparse
import ctypes
import mmap
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from kvcr import KVCR, KVCRBindings
from kvcr.config import G3Options, KVCRBackendConfigs, KVCRConfig, LocalDramOptions
from kvcr.types import BlockKey, CacheTier, MemoryRef, QueryStatus, RegionDescriptor


def profile(name):
    if name == "synthetic":
        return {"full": 2 * mmap.PAGESIZE, "swa": mmap.PAGESIZE}, [
            "full",
            "swa:a",
            "swa:b",
        ]

    # Cache-only Kimi K3 configuration, TP=1, 64-token MLA blocks, no speculative
    # decoding or RecoverSSM. No model code or weights are downloaded.
    # https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json
    import torch
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    mla = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512 + 64,
        dtype=torch.bfloat16,
    )
    kda = MambaSpec(
        block_size=64,
        shapes=MambaStateShapeCalculator.kda_state_shape(
            1, 96, 128, conv_kernel_size=4, num_spec=0
        ),
        dtypes=MambaStateDtypeCalculator.kda_state_dtype(
            torch.bfloat16, "auto", "auto"
        ),
    )
    print("Kimi K3 cache-only profile: TP=1, BF16 MLA, BF16 conv / FP32 recurrent.")
    return {"mla": mla.page_size_bytes, "kda": kda.page_size_bytes}, ["mla", "kda"]


def wait(kvcr, handle):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for completed, entries in kvcr.poll_completed():
            if completed == handle:
                if not all(entry.success for entry in entries.values()):
                    raise RuntimeError(f"Operation {handle} failed: {entries}")
                return entries
        time.sleep(0.001)
    raise TimeoutError(f"Operation {handle} did not complete")


def refs(index, layout):
    counts = Counter(label.partition(":")[0] for label in layout)
    offsets = Counter()
    result = []
    for label in layout:
        pool = label.partition(":")[0]
        result.append(
            MemoryRef(
                end_point_name="hybrid-g3-demo",
                label=label,
                element_index=index * counts[pool] + offsets[pool],
            )
        )
        offsets[pool] += 1
    return result


def run(directory, backend, profile_name="synthetic"):
    sizes, layout = profile(profile_name)
    counts = Counter(label.partition(":")[0] for label in layout)
    key_bytes = sum(sizes[label.partition(":")[0]] for label in layout)
    with ExitStack() as stack:
        # Anonymous mmaps provide page-aligned buffers for direct file I/O.
        primary = {
            name: stack.enter_context(mmap.mmap(-1, 3 * counts[name] * size))
            for name, size in sizes.items()
        }
        local = {
            name: stack.enter_context(mmap.mmap(-1, counts[name] * size))
            for name, size in sizes.items()
        }

        def address(buffer):
            return ctypes.addressof(ctypes.c_char.from_buffer(buffer))

        def no_pin(_keys):
            raise RuntimeError("This local-only demo should not request peer pins")

        kvcr = KVCR(
            KVCRConfig(
                nixl_agent_name="hybrid-g3-demo",
                pool_layouts=list(sizes.items()),
                nixl_listen_port=0,
                operation_timeout_ms=10000,
                abandon_timeout_ms=20000,
            ),
            KVCRBindings(
                request_pin=no_pin,
                poll_pin_results=lambda: (),
                release_pin=lambda _handle: True,
            ),
            KVCRBackendConfigs(
                framework_regions=[
                    RegionDescriptor(
                        addr=address(primary[name]),
                        size=size,
                        count=len(primary[name]) // size,
                        label=f"{name}:*"
                        if any(label.startswith(f"{name}:") for label in layout)
                        else name,
                    )
                    for name, size in sizes.items()
                ],
                local_dram=LocalDramOptions(
                    [(name, address(buf), len(buf)) for name, buf in local.items()]
                ),
                g3=G3Options(
                    paths=(directory / "hybrid.data",),
                    capacity_bytes_per_file=2 * key_bytes,
                    backend=backend,
                ),
            ),
        )
        stack.callback(kvcr.close)
        expected = {}
        for key_index in range(2):
            for piece_index, ref in enumerate(refs(key_index, layout)):
                pool = ref.label.partition(":")[0]
                size = sizes[pool]
                data = bytes([key_index * 16 + piece_index + 1]) * size
                offset = ref.element_index * size
                primary[pool][offset : offset + size] = data
                expected[key_index, ref.label, piece_index] = data
        first, second = BlockKey(b"first"), BlockKey(b"second")
        print(f"Backend: {backend}; file: {directory / 'hybrid.data'}", flush=True)
        print(f"Profile: {profile_name}; pool bytes={sizes}; layout={layout}")
        wait(kvcr, kvcr.deposit({first: refs(0, layout)}))
        wait(kvcr, kvcr.deposit({second: refs(1, layout)}))
        assert kvcr.query([first]) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
        print("1. Deposited two keys: first spilled from DRAM to G3.", flush=True)

        wait(kvcr, kvcr.deliver({first: refs(2, layout)}))
        for index, ref in enumerate(refs(2, layout)):
            pool = ref.label.partition(":")[0]
            offset = ref.element_index * sizes[pool]
            assert (
                primary[pool][offset : offset + sizes[pool]]
                == expected[0, ref.label, index]
            )
        print("2. Read G3 into primary buffers: every piece matches.", flush=True)

        handle = kvcr.fetch([first], expected_layout=layout)
        result = wait(kvcr, handle)[first]
        for index, ref in enumerate(result.descriptors):
            pool = ref.label.partition(":")[0]
            offset = ref.element_index * sizes[pool]
            assert (
                local[pool][offset : offset + sizes[pool]]
                == expected[0, ref.label, index]
            )
        kvcr.release([result.release_handle])
        print("3. Refilled DRAM pools from G3: all bytes match.", flush=True)
        print("PASS: multi-pool DRAM -> G3 -> primary / DRAM round trip.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="GDS_MT")
    parser.add_argument(
        "--profile", choices=("synthetic", "kimi-k3"), default="synthetic"
    )
    parser.add_argument("--directory", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    # Use fresh files, so this demo never alters another KVCR's disk cache.
    with TemporaryDirectory(prefix="kvcr-multi-pool-", dir=args.directory) as temp:
        run(Path(temp), args.backend, args.profile)
