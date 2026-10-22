# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allocated-GPU-only two-process KVCR CUDA IPC roundtrip.

Run explicitly; this is not a local unit test. Payload initialization and complete
byte comparisons stay on CUDA. Only control metadata and scalar oracle results
return to the host. A byte PASS alone is not hidden-host-staging trace proof.
"""

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
import uuid
from pathlib import Path

SOURCE_SLOTS = (11, 2, 8, 4, 1, 10, 6)
TARGET_SLOTS = (0, 12, 3, 9, 5, 1, 7)
SLOTS = 13


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_source(manifest):
    expected = json.loads(Path(manifest).read_text())["package_entries_sha256"]
    spec = importlib.util.find_spec("kvcr")
    assert spec is not None and spec.origin is not None
    package = Path(spec.origin).resolve().parent
    actual = {
        "src/kvcr/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in package.glob("*.py")
    }
    assert actual == expected, (
        "Actual imported KVCR source differs from external manifest"
    )
    return {"package": str(package), "package_entries_sha256": actual}


class Stats:
    def __init__(self):
        self.records = []

    def increase_counter(self, name, value, labelvalues=()):
        self.records.append(["counter", name, value, list(labelvalues)])

    def set_gauge(self, name, value, labelvalues=()):
        self.records.append(["gauge", name, value, list(labelvalues)])

    def observe_histogram(self, name, value, labelvalues=()):
        self.records.append(["histogram", name, value, list(labelvalues)])

    def reduce(self):
        return {}

    def is_empty(self):
        return not self.records


class Keys:
    def decode(self, key):
        return int.from_bytes(key, "big")


def gpu_identity(device):
    """Read the current process's actual CUDA device UUID/PCI metadata."""
    import ctypes

    class CUuuid(ctypes.Structure):
        _fields_ = [("value", ctypes.c_ubyte * 16)]

    driver = ctypes.CDLL("libcuda.so.1")
    ordinal = ctypes.c_int()
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    assert driver.cuDeviceGet(ctypes.byref(ordinal), device) == 0
    identifier = CUuuid()
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
    get_uuid.argtypes = [ctypes.POINTER(CUuuid), ctypes.c_int]
    get_uuid.restype = ctypes.c_int
    assert get_uuid(ctypes.byref(identifier), ordinal.value) == 0
    pci = ctypes.create_string_buffer(64)
    driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
    assert driver.cuDeviceGetPCIBusId(pci, len(pci), ordinal.value) == 0
    return {
        "uuid": str(uuid.UUID(bytes=bytes(identifier.value))),
        "pci_bus_id": pci.value.decode("ascii"),
        "cuda_device": ordinal.value,
    }


def mapped_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if (
            len(fields) == 6
            and fields[5].startswith("/")
            and any(
                token in fields[5]
                for token in (
                    "libplugin_UCX",
                    "libucp",
                    "libucs",
                    "libnixl",
                    "libserdes",
                    "_bindings",
                )
            )
        ):
            paths.add(fields[5])
    assert len([p for p in paths if "libplugin_UCX" in p]) == 1
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}


def _peer(
    role,
    device,
    row_bytes,
    control_port,
    nixl_port,
    source_endpoint,
    domain,
    run_id,
    output,
    manifest,
    profile_enabled,
    ready,
    stop,
    channel,
):
    output = Path(output)
    kvcr = None
    bank = None
    profiler = None
    profile_range = None
    profile_info = {"requested": profile_enabled, "trace_qualified": False}
    nvtx_active = False

    def stop_profile():
        nonlocal profiler, profile_range, nvtx_active
        if profiler is None:
            return
        if nvtx_active:
            torch.cuda.nvtx.range_pop()
            nvtx_active = False
        if profile_range is not None:
            profile_range.__exit__(None, None, None)
            profile_range = None
        captured = profiler
        profiler = None
        captured.stop()
        profile_end_ns = time.time_ns()
        trace = output / f"{role}.cuda-trace.json"
        captured.export_chrome_trace(str(trace))
        profile_info.update(
            {
                "status": "TRACE_CAPTURED_UNREVIEWED",
                "end_unix_ns": profile_end_ns,
                "trace_file": trace.name,
                "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            }
        )

    try:
        assert os.environ.get("UCX_TLS") == "cuda_ipc,posix,self"
        assert os.environ.get("NIXL_UCX_CUDA_IPC_ONLY") == "1"
        source_proof = verify_source(manifest)
        import torch

        from kvcr import KVCR, KVCRBindings
        from kvcr.config import KVCRBackendConfigs, KVCRConfig, RemoteFWGPUOptions
        from kvcr.control_channels import ZmqPeerControlChannel
        from kvcr.types import BlockKey, MemoryRef, PinRequestId, RegionDescriptor

        torch.cuda.set_device(device)
        physical_device = gpu_identity(device)
        bank = torch.full((SLOTS, row_bytes), 255, dtype=torch.uint8, device=device)
        pattern_index = torch.arange(row_bytes, dtype=torch.int64, device=device)

        def pattern(key):
            value = (pattern_index ^ ((key + 1) * 0x9E3779B1)) & 0xFFFFFFFF
            value = ((value ^ (value >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
            value = ((value ^ (value >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
            return (value ^ (value >> 16)).to(torch.uint8)

        if role == "source":
            for key, slot in enumerate(SOURCE_SLOTS):
                bank[slot].copy_(pattern(key))
        torch.cuda.synchronize(device)
        name = f"kvcr-gpu-{run_id}-{role}"
        pending = []
        pin_requests, releases, hazards = [], [], []
        keys = tuple(BlockKey(i.to_bytes(8, "big")) for i in range(7))

        def request_pin(requested):
            assert role == "source"
            request = PinRequestId(len(pin_requests))
            pin_requests.append([bytes(key).hex() for key in requested])
            # Producer initialization was synchronized before exposing addresses.
            pending.append(
                (
                    request,
                    (
                        "pin",
                        {
                            key: [
                                MemoryRef(
                                    end_point_name=name,
                                    element_index=SOURCE_SLOTS[
                                        int.from_bytes(key, "big")
                                    ],
                                )
                            ]
                            if key in keys
                            else None
                            for key in requested
                        },
                    ),
                )
            )
            return request

        def poll_pin():
            result = pending[:]
            del pending[: len(result)]
            return result

        def release_pin(handle):
            releases.append(handle)
            return True

        def hazard(error):
            hazards.append(
                {
                    "type": type(error).__name__,
                    "state": getattr(error, "state", None),
                    "operation": getattr(error, "op_handle", None),
                }
            )

        region = RegionDescriptor(
            mem_type="VRAM",
            device_Id=device,
            addr=bank.data_ptr(),
            size=row_bytes,
            stride=bank.stride(0),
            count=SLOTS,
        )
        kvcr = KVCR(
            KVCRConfig(
                nixl_agent_name=name,
                pool_layouts=[("", row_bytes)],
                operation_timeout_ms=30000,
                abandon_timeout_ms=60000,
                startup_timeout_seconds=60,
                nixl_listen_port=nixl_port,
                enable_telemetry=True,
            ),
            KVCRBindings(
                request_pin=request_pin,
                poll_pin_results=poll_pin,
                release_pin=release_pin,
                key_adapter=Keys(),
                stats_factory=Stats,
                on_resilience_event=hazard,
                framework_control=ZmqPeerControlChannel(
                    "127.0.0.1", control_port, "127.0.0.1"
                ),
            ),
            KVCRBackendConfigs(
                framework_regions=[region],
                remote_fw_gpu=RemoteFWGPUOptions(
                    compatibility_domain=domain, eager_ctrl_connect=False
                ),
            ),
        )
        proof = {
            "role": role,
            "pid": os.getpid(),
            "device": device,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "source": source_proof,
            "registration": {
                "memory_type": region.mem_type,
                "device_Id": region.device_Id,
                "address": region.addr,
                "stride": region.stride,
                "count": region.count,
                "element_bytes": region.size,
            },
            "configured_cpu_or_disk_payload_pools": 0,
            "gpu_pattern": "mixed32-position-key-v1",
            "UCX_TLS": os.environ["UCX_TLS"],
            "NIXL_UCX_CUDA_IPC_ONLY": "1",
            "NIXL_PLUGIN_DIR": os.environ.get("NIXL_PLUGIN_DIR"),
        }
        proof["physical_device"] = physical_device
        proof["native_libraries"] = mapped_libraries()
        proof["profile"] = profile_info
        if profile_enabled:
            assert (
                torch.profiler.ProfilerActivity.CUDA
                in torch.profiler.supported_activities()
            ), "CUDA profiler activity unsupported"
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            profiler.start()
            profile_info["start_unix_ns"] = time.time_ns()
            profile_range = torch.profiler.record_function(
                f"KVCR_GPU_{role}_PAYLOAD_WINDOW"
            )
            profile_range.__enter__()
            try:
                torch.cuda.nvtx.range_push(f"KVCR_GPU_{role}_PAYLOAD_WINDOW")
                nvtx_active = True
                profile_info["nvtx_range_pushed"] = True
            except RuntimeError:
                profile_info["nvtx_range_pushed"] = False
        write(output / f"{role}.ready.json", proof)
        ready.set()
        deadline = time.monotonic() + (240 if role == "source" else 150)
        result = []
        if role == "target":
            assert stop.wait(0) is False
            kvcr.submit_hint(
                {
                    "protocol_version": "0.1",
                    "actions": [
                        {
                            "action_type": "kv.fetch_gpu",
                            "action_version": "1.0",
                            "payload": {
                                "source_control_endpoint": source_endpoint,
                                "block_hashes": list(range(7)),
                                "source_memory_kind": "gpu",
                                "target_memory_kind": "gpu",
                                "transport": "nixl_gpu_direct",
                                "compatibility_domain": domain,
                            },
                        }
                    ],
                },
                "roundtrip",
            )
            handle = kvcr.deliver(
                {
                    key: [MemoryRef(end_point_name=name, element_index=slot)]
                    for key, slot in zip(keys, TARGET_SLOTS, strict=True)
                },
                "roundtrip",
            )
            while not result and time.monotonic() < deadline:
                result.extend(kvcr.poll_completed())
                time.sleep(0.001)
            assert len(result) == 1 and result[0][0] == handle
            assert set(result[0][1]) == set(keys) and all(
                e.success for e in result[0][1].values()
            )
            torch.cuda.synchronize(device)
            stop_profile()  # Exclude verification scalar DtoH copies.
            mismatch_by_key = {
                str(i): int(torch.count_nonzero(bank[slot] != pattern(i)).item())
                for i, slot in enumerate(TARGET_SLOTS)
            }
            assert not any(mismatch_by_key.values()), mismatch_by_key
            untouched = [i for i in range(SLOTS) if i not in TARGET_SLOTS]
            assert all(bool(torch.all(bank[i] == 255).item()) for i in untouched)
            proof.update(
                {
                    "full_gpu_byte_oracle": "PASS",
                    "mismatch_bytes_by_key": mismatch_by_key,
                    "verified_payload_bytes": row_bytes * 7,
                    "target_slots": TARGET_SLOTS,
                    "untouched_guard_slots": untouched,
                    "completed_keys": 7,
                }
            )
        else:
            while not stop.is_set() and time.monotonic() < deadline:
                assert not list(kvcr.poll_completed())
                time.sleep(0.001)
            assert stop.is_set(), "Target never signaled completion"
            while kvcr._core._framework_pin_keys and time.monotonic() < deadline:
                list(kvcr.poll_completed())
                time.sleep(0.001)
            assert pin_requests == [[bytes(k).hex() for k in keys]] and releases == [
                "pin"
            ]
            assert not kvcr._core._framework_pin_keys
            stop_profile()  # Native source work and pin release have completed.
            proof.update(
                {
                    "source_slots": SOURCE_SLOTS,
                    "pin_requests": pin_requests,
                    "releases": releases,
                    "source_native_pins_drained": True,
                }
            )
        list(kvcr.poll_completed())
        assert not hazards, hazards
        stats = kvcr.get_stats()
        proof.update({"telemetry": stats.records, "hazards": hazards})
        kvcr.close()
        assert kvcr._core.is_quiescent()
        kvcr = None
        proof.update({"status": "PASS_ACTUAL_KVCR_GPU_PEER", "close_quiescent": True})
        write(output / f"{role}.result.json", proof)
        channel.send({"status": "PASS", "role": role})
    except BaseException:
        error = {
            "status": "FAIL_ACTUAL_KVCR_GPU_PEER",
            "role": role,
            "pid": os.getpid(),
            "traceback": traceback.format_exc(),
        }
        write(output / f"{role}.failure.json", error)
        channel.send(error)
    finally:
        if profiler is not None:
            try:
                stop_profile()
            except BaseException:
                write(
                    output / f"{role}.profile-failure.json",
                    {"traceback": traceback.format_exc()},
                )
        if kvcr is not None:
            try:
                kvcr.close()
            except BaseException:
                write(
                    output / f"{role}.cleanup-failure.json",
                    {"traceback": traceback.format_exc()},
                )
        # Keep bank owned until the native close attempt and then process exit.
        _ = bank
        channel.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-device", type=int, default=0)
    parser.add_argument("--target-device", type=int, default=1)
    parser.add_argument(
        "--row-bytes", type=int, choices=(9437184, 17694720), required=True
    )
    parser.add_argument("--port-base", type=int, default=28340)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture child CUDA activity; independent trace review required",
    )
    args = parser.parse_args()
    assert args.source_device != args.target_device
    args.output.mkdir(parents=True, exist_ok=False)
    # Reserve/probe only four explicit loopback control/metadata ports, no scans.
    sockets = []
    for offset in range(4):
        s = socket.socket()
        s.bind(("127.0.0.1", args.port_base + offset))
        sockets.append(s)
    for s in sockets:
        s.close()
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    ready = [ctx.Event(), ctx.Event()]
    parents, children, processes = [], [], []
    run_id = uuid.uuid4().hex
    domain = "same-host-cuda-ipc:" + run_id
    try:
        for i, (role, device) in enumerate(
            (("source", args.source_device), ("target", args.target_device))
        ):
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=_peer,
                args=(
                    role,
                    device,
                    args.row_bytes,
                    args.port_base + i,
                    args.port_base + 2 + i,
                    f"tcp://127.0.0.1:{args.port_base}",
                    domain,
                    run_id,
                    str(args.output.resolve()),
                    str(args.source_manifest.resolve()),
                    args.profile,
                    ready[i],
                    stop,
                    child,
                ),
            )
            parents.append(parent)
            children.append(child)
            processes.append(p)
            p.start()
            child.close()
            startup_deadline = time.monotonic() + 75
            while not ready[i].is_set() and time.monotonic() < startup_deadline:
                if parent.poll(0.05):
                    raise RuntimeError(parent.recv())
            assert ready[i].is_set(), f"{role} failed native startup"
        assert parents[1].poll(155), "Target deadline exceeded"
        target = parents[1].recv()
        assert target["status"] == "PASS", target
        stop.set()
        assert parents[0].poll(30), "Source release deadline exceeded"
        source = parents[0].recv()
        assert source["status"] == "PASS", source
        for p in processes:
            p.join(15)
            assert p.exitcode == 0
        physical = [
            json.loads((args.output / f"{role}.result.json").read_text())[
                "physical_device"
            ]
            for role in ("source", "target")
        ]
        assert physical[0]["uuid"] != physical[1]["uuid"], (
            "Peers did not use distinct physical GPU UUIDs"
        )
        assert physical[0]["pci_bus_id"] != physical[1]["pci_bus_id"], (
            "Peers must use distinct physical GPU PCI devices"
        )
        write(
            args.output / "result.json",
            {
                "status": "PASS_REAL_KVCR_GPU_CUDA_IPC_BYTE_ROUNDTRIP",
                "row_bytes": args.row_bytes,
                "copied_keys": 7,
                "copied_payload_bytes": args.row_bytes * 7,
                "source_device": args.source_device,
                "target_device": args.target_device,
                "physical_devices": physical,
                "profile_requested": args.profile,
                "verification": "FULL_BYTE_GPU_ORACLE_NO_HOST_PAYLOAD_DUMP",
                "physical_no_host_staging_trace_qualified": False,
                "limits": (
                    "Actual KVCR/VRAM byte proof; separate CUDA/CUPTI trace "
                    "required to certify hidden backend staging. "
                    "No model or vLLM layout claim."
                ),
            },
        )
    finally:
        stop.set()
        for p in processes:
            p.join(5)
            if p.is_alive():
                p.terminate()
                p.join(5)
            if p.is_alive():
                p.kill()
                p.join(5)
        for parent in parents:
            parent.close()


if __name__ == "__main__":
    main()
