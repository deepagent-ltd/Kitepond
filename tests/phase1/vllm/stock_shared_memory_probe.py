# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify two native CUDA mappings of the same GLM FP8 rank-local host pages."""

import hashlib
import json
import mmap
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from kitepond.models.glm5next.region_socket import RegionPublisher, join_region
from kitepond.models.glm5next.shared_region import NativeHiSparseMemfdRegion

from vllm.v1.hisparse.runtime import release_pinned_state
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

ROWS = 12
ROW_WIDTH = 656
PAYLOAD = ROWS * ROW_WIDTH
LAYOUT_SHA = hashlib.sha256(b"GLM SM120 FP8-DCP4 656 byte rank local").hexdigest()
IDENTITY = {
    "authority_engine_id": "decode-microprobe",
    "authority_boot_id": "epoch-1",
    "source_rank": 0,
    "layout_sha256": LAYOUT_SHA,
}


def _canary():
    return (torch.arange(PAYLOAD, dtype=torch.int32) % 251).to(torch.uint8)


def _unpin(tensor):
    pins = [tensor]
    release_pinned_state([], pins)
    if pins:
        raise RuntimeError("Native CUDA host unregister did not finish")


def _producer(directory, sender):
    with sender:
        region = join_region(directory, **IDENTITY)
        tensor = region.tensor()
        pin_tensor(tensor)
        try:
            device = tensor[:PAYLOAD].to("cuda", non_blocking=True)
            assert torch.equal(device.cpu(), _canary())
            tensor[ROW_WIDTH : 2 * ROW_WIDTH].copy_(
                torch.full((ROW_WIDTH,), 17, dtype=torch.uint8, device="cuda"),
                non_blocking=True,
            )
            torch.accelerator.synchronize()
            sender.send(
                {
                    "inode": region.descriptor.inode,
                    "device_id": region.descriptor.device_id,
                }
            )
        finally:
            _unpin(tensor)
            del tensor
            region.close()


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Run the proof on exactly one SM120 GPU")
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    extent = (PAYLOAD // mmap.PAGESIZE + 1) * mmap.PAGESIZE
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        region = NativeHiSparseMemfdRegion.create(extent, **IDENTITY)
        publisher = RegionPublisher(directory, region)
        tensor = region.tensor()
        tensor.fill_(239)
        tensor[:PAYLOAD].copy_(_canary())
        pin_tensor(tensor)
        proc = ctx.Process(target=_producer, args=(directory, sender))
        try:
            proc.start()
            sender.close()
            assert publisher.serve_one(accept_timeout=30)
            assert receiver.poll(30)
            peer = receiver.recv()
            assert peer["inode"] == region.descriptor.inode
            assert peer["device_id"] == region.descriptor.device_id
            expected = _canary()
            expected[ROW_WIDTH : 2 * ROW_WIDTH].fill_(17)
            observed = tensor[:PAYLOAD].to("cuda", non_blocking=True).cpu()
            assert torch.equal(observed, expected)
            assert torch.all(tensor[PAYLOAD:] == 239).item()
            proc.join(30)
            assert proc.exitcode == 0
            print(
                json.dumps(
                    {
                        "device": "SM120",
                        "native_registered_processes": 2,
                        "native_same_inode": True,
                        "row_bytes": ROW_WIDTH,
                        "checked_bytes": PAYLOAD,
                        "physical_host_extent_bytes": extent,
                        "guards_unchanged": True,
                    }
                )
            )
        finally:
            if proc.pid is not None and proc.is_alive():
                proc.terminate()
                proc.join(5)
            receiver.close()
            sender.close()
            publisher.close()
            _unpin(tensor)
            del tensor
            region.close()


if __name__ == "__main__":
    main()
