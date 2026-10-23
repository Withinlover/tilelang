"""Stable metadata and address-free descriptors for the local P0 model.

This encoding is deliberately not a hardware communication ABI. Its zero
address fields cannot be passed to a device transport.
"""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

LOCAL_PROFILE = "ascend950_cannsim_v1"
NATIVE_PROFILE = "ascend950_shmem_mte_phased_v1"
MANIFEST_ATTR = "tl.distributed.manifest"
DESCRIPTOR_HEADER_BYTES = 64
DESCRIPTOR_RECORD_BYTES = 32
SIGNAL_STRIDE_BYTES = 64
MAX_EPOCH = (1 << 31) - 1


def descriptor_nbytes(object_count: int, world_size: int) -> int:
    return DESCRIPTOR_HEADER_BYTES + object_count * world_size * DESCRIPTOR_RECORD_BYTES


def canonical_manifest(manifest) -> dict:
    """Copy serializable metadata, rejecting NaNs and process-local objects."""
    if isinstance(manifest, str):
        manifest = json.loads(manifest)
    if not isinstance(manifest, dict):
        raise TypeError("distributed manifest must be a dictionary or JSON object")
    return json.loads(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False))


def manifest_digest(manifest) -> str:
    encoded = json.dumps(canonical_manifest(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def kernel_manifest(kernel) -> dict:
    manifest = getattr(kernel, "manifest", None)
    if manifest is None:
        func = getattr(kernel, "prim_func", kernel)
        attrs = getattr(func, "attrs", None)
        if attrs is not None:
            manifest = attrs.get(MANIFEST_ATTR)
    if manifest is None:
        raise ValueError("kernel has no canonical P0 distributed manifest")
    return canonical_manifest(str(manifest) if not isinstance(manifest, (dict, str)) else manifest)


def encode_descriptor(manifest, *, rank, generation, launch_id, epoch_base, epoch_end) -> np.ndarray:
    """Create a CPU-only POD record without tensor or process addresses."""
    manifest = canonical_manifest(manifest)
    groups, objects = manifest["groups"], manifest["objects"]
    world_size = groups[0]["size"]
    expected_size = descriptor_nbytes(len(objects), world_size)
    descriptor = manifest["descriptor"]
    if descriptor.get("abi_version") != 1 or descriptor.get("nbytes") != expected_size:
        raise ValueError("distributed descriptor ABI or size does not match the local profile")
    if descriptor.get("encoding", "cpu-model-v1") != "cpu-model-v1":
        raise ValueError("only the address-free cpu-model-v1 descriptor is supported")
    if not 0 <= rank < world_size or not 0 <= epoch_base < epoch_end <= MAX_EPOCH:
        raise ValueError("invalid rank or positive int32 epoch range in descriptor")
    raw = bytearray(expected_size)
    struct.pack_into(
        "<8sIIIIQQQQQ", raw, 0, b"TDCOSIM\0", 1, rank, world_size, len(objects), generation, launch_id, epoch_base, epoch_end, 0
    )
    offset = DESCRIPTOR_HEADER_BYTES
    for index, obj in enumerate(objects):
        flags = {"read": 1, "write": 2, "read_write": 3}[obj["remote_access"]]
        if obj["kind"] == "signal":
            flags |= 4
        for owner in range(world_size):
            struct.pack_into("<IIQQII", raw, offset, index, owner, 0, obj["physical_nbytes"], flags, 0)
            offset += DESCRIPTOR_RECORD_BYTES
    result = np.frombuffer(bytes(raw), dtype=np.uint8)
    return result
