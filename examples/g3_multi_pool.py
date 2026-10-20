# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise native NIXL multi-pool G3 transfers without downloading a model."""

import argparse
import ctypes
import mmap
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from kvcr import KVCR, KVCRBindings
from kvcr.config import G3Options, KVCRBackendConfigs, KVCRConfig, LocalDramOptions
from kvcr.types import BlockKey, CacheTier, MemoryRef, QueryStatus, RegionDescriptor

LAYOUT = ["full", "swa:a", "swa:b"]


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


def refs(index):
    return [
        MemoryRef(
            end_point_name="hybrid-g3-demo", label=LAYOUT[0], element_index=index
        ),
        MemoryRef(
            end_point_name="hybrid-g3-demo", label=LAYOUT[1], element_index=2 * index
        ),
        MemoryRef(
            end_point_name="hybrid-g3-demo",
            label=LAYOUT[2],
            element_index=2 * index + 1,
        ),
    ]


def run(directory, backend):
    page = mmap.PAGESIZE
    sizes = {"full": 2 * page, "swa": page}
    with ExitStack() as stack:
        # Anonymous mmaps provide page-aligned buffers for direct file I/O.
        primary = {name: stack.enter_context(mmap.mmap(-1, 6 * page)) for name in sizes}
        local = {name: stack.enter_context(mmap.mmap(-1, 2 * page)) for name in sizes}

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
                        label="swa:*" if name == "swa" else name,
                    )
                    for name, size in sizes.items()
                ],
                local_dram=LocalDramOptions(
                    [(name, address(buf), len(buf)) for name, buf in local.items()]
                ),
                g3=G3Options(
                    paths=(directory / "hybrid.data",),
                    capacity_bytes_per_file=8 * page,
                    backend=backend,
                ),
            ),
        )
        stack.callback(kvcr.close)
        primary["full"][: 4 * page] = b"f" * (2 * page) + b"F" * (2 * page)
        primary["swa"][: 4 * page] = (
            b"a" * page + b"b" * page + b"A" * page + b"B" * page
        )
        first, second = BlockKey(b"first"), BlockKey(b"second")
        print(f"Backend: {backend}; file: {directory / 'hybrid.data'}", flush=True)
        print(f"Pools: full={2 * page} bytes, swa={page} bytes; layout={LAYOUT}")
        wait(kvcr, kvcr.deposit({first: refs(0)}))
        wait(kvcr, kvcr.deposit({second: refs(1)}))
        assert kvcr.query([first]) == [(QueryStatus.FETCHABLE, CacheTier.G3)]
        print("1. Deposited two keys: first spilled from DRAM to G3.", flush=True)

        wait(kvcr, kvcr.deliver({first: refs(2)}))
        assert primary["full"][4 * page :] == b"f" * (2 * page)
        assert primary["swa"][4 * page :] == b"a" * page + b"b" * page
        print("2. Read G3 into primary buffers: all three pieces match.", flush=True)

        handle = kvcr.fetch([first], expected_layout=LAYOUT)
        result = wait(kvcr, handle)[first]
        assert local["full"][:] == b"f" * (2 * page)
        assert local["swa"][:] == b"a" * page + b"b" * page
        kvcr.release([result.release_handle])
        print("3. Refilled both DRAM pools from G3: all bytes match.", flush=True)
        print("PASS: multi-pool DRAM -> G3 -> primary / DRAM round trip.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="GDS_MT")
    parser.add_argument("--directory", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    # Use fresh files, so this demo never alters another KVCR's disk cache.
    with TemporaryDirectory(prefix="kvcr-multi-pool-", dir=args.directory) as temp:
        run(Path(temp), args.backend)
