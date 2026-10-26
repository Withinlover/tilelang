"""A single CPU coordinator owns all ranks of a local P0 simulation.

There is no process group, device pointer registration, or certified hardware
transport in this profile. Symmetric BF16 storage contains raw uint16 bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import secrets
from threading import RLock

import numpy as np

from .abi import (
    LOCAL_PROFILE,
    MAX_EPOCH,
    NATIVE_PROFILE,
    SIGNAL_STRIDE_BYTES,
    canonical_manifest,
    encode_descriptor,
    kernel_manifest,
    manifest_digest,
)


class UnsupportedProfileError(ValueError):
    """The requested transport or execution environment is not certified."""


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _storage_dtype(dtype):
    if str(dtype) == "bfloat16":
        return np.dtype("<u2")
    supported = {"float16", "float32", "float64", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}
    if str(dtype) not in supported:
        raise ValueError(f"unsupported symmetric dtype: {dtype}")
    return np.dtype(str(dtype)).newbyteorder("<")


def _contiguous_strides(shape):
    stride, result = 1, []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _aligned_array(shape, dtype, *, signal=False):
    item_stride = SIGNAL_STRIDE_BYTES if signal else dtype.itemsize
    backing = np.zeros(math.prod(shape) * item_stride + 63, dtype=np.uint8)
    offset = (-backing.ctypes.data) % 64
    strides = tuple(s * item_stride for s in _contiguous_strides(shape))
    return np.ndarray(shape, dtype=dtype, buffer=backing, offset=offset, strides=strides)


@dataclass(eq=False)
class SymmetricBuffer:
    session: Session
    key: str
    shape: tuple[int, ...]
    dtype: str
    remote_access: str
    kind: str
    arrays: dict[int, np.ndarray]

    @property
    def physical_nbytes(self):
        return math.prod(self.shape) * (SIGNAL_STRIDE_BYTES if self.kind == "signal" else _storage_dtype(self.dtype).itemsize)

    @property
    def strides(self):
        return tuple(s // self.arrays[0].dtype.itemsize for s in self.arrays[0].strides)

    def for_rank(self, rank):
        return self.arrays[rank]


@dataclass(frozen=True, eq=False)
class Channel:
    session: Session
    key: str
    peers: dict[int, tuple[int, ...]]


@dataclass(frozen=True, eq=False)
class LaunchTicket:
    session: Session
    kernel: object
    channel: Channel
    generation: int
    launch_id: int
    base: int
    end: int
    protocol: str
    parameters: dict
    manifest: dict
    manifest_digest: str
    descriptors: dict[int, np.ndarray]

    @property
    def epochs(self):
        return self.end - self.base


@dataclass(eq=False)
class Session:
    world_size: int
    profile: str = LOCAL_PROFILE
    generation: int = field(default_factory=lambda: secrets.randbits(63) + 1)
    state: str = "QUIESCENT"
    objects: dict[str, SymmetricBuffer] = field(default_factory=dict)
    channels: dict[str, Channel] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, repr=False)
    _epoch: int = 0
    _launch_id: int = 0
    _ticket: LaunchTicket | None = None
    _run: object = None


def init(*, process_group=None, device=None, profile=LOCAL_PROFILE, world_size=2, **options):
    if profile == NATIVE_PROFILE:
        from tilelang.ascend.distributed.native_runtime import init_runtime

        return init_runtime(profile=profile, process_group=process_group, device=device, world_size=world_size, **options)
    if profile != LOCAL_PROFILE:
        raise UnsupportedProfileError(
            f"profile {profile!r} is not certified; select {LOCAL_PROFILE!r} for local functional co-simulation "
            f"or {NATIVE_PROFILE!r} for explicit native SHMEM phases"
        )
    if process_group is not None or device is not None:
        raise UnsupportedProfileError("the local profile uses one CPU coordinator; process_group and device must be None")
    if options:
        raise TypeError(f"unknown local session options: {sorted(options)}")
    if _positive_int(world_size, "world_size") != 2:
        raise UnsupportedProfileError("the local P0 profile currently supports exactly two logical ranks")
    return Session(world_size=world_size, profile=profile)


def _allocate(ctx, *, key, shape, dtype, remote_access, kind):
    with ctx._lock:
        if ctx.state != "QUIESCENT":
            raise RuntimeError("symmetric allocation requires a QUIESCENT session")
        if not isinstance(key, str) or not key or key in ctx.objects:
            raise ValueError("symmetric object keys must be nonempty and unique")
        shape = tuple(_positive_int(dim, "shape extent") for dim in shape)
        if not shape:
            raise ValueError("symmetric shape must have at least one dimension")
        if remote_access not in ("read", "write", "read_write"):
            raise ValueError("remote_access must be read, write, or read_write")
        storage_dtype = _storage_dtype(dtype)
        obj = SymmetricBuffer(
            ctx,
            key,
            shape,
            str(dtype),
            remote_access,
            kind,
            {rank: _aligned_array(shape, storage_dtype, signal=kind == "signal") for rank in range(ctx.world_size)},
        )
        ctx.objects[key] = obj
        return obj


def alloc_symmetric(ctx, *, key, shape, dtype, remote_access="read_write"):
    if ctx.profile == NATIVE_PROFILE:
        return ctx.allocate(key=key, shape=shape, dtype=dtype, remote_access=remote_access, kind="symmetric")
    return _allocate(ctx, key=key, shape=shape, dtype=dtype, remote_access=remote_access, kind="symmetric")


def alloc_signals(ctx, *, key, shape, dtype="int32"):
    if ctx.profile == NATIVE_PROFILE:
        return ctx.allocate(key=key, shape=shape, dtype=dtype, remote_access="write", kind="signal")
    if str(dtype) != "int32":
        raise UnsupportedProfileError("the local P0 profile freezes SignalValue to int32")
    return _allocate(ctx, key=key, shape=shape, dtype=dtype, remote_access="write", kind="signal")


def channel(ctx, *, key, peers=None):
    if ctx.profile == NATIVE_PROFILE:
        return ctx.channel(key=key, peers=peers)
    with ctx._lock:
        if ctx.state != "QUIESCENT":
            raise RuntimeError("channel creation requires a QUIESCENT session")
        if not isinstance(key, str) or not key or key in ctx.channels:
            raise ValueError("channel keys must be nonempty and unique")
        if ctx.channels:
            raise UnsupportedProfileError("P0 permits one channel per session")
        if peers is None:
            peers = {rank: tuple(other for other in range(ctx.world_size) if other != rank) for rank in range(ctx.world_size)}
        if not isinstance(peers, dict) or set(peers) != set(range(ctx.world_size)):
            raise ValueError("peers must map every logical rank to its non-self peer ranks")
        normalized = {rank: tuple(values) for rank, values in peers.items()}
        for rank, values in normalized.items():
            if (
                not values
                or len(values) != len(set(values))
                or any(isinstance(p, bool) or not isinstance(p, int) or p == rank or not 0 <= p < ctx.world_size for p in values)
            ):
                raise ValueError("channel peers must be unique, in-group, non-self ranks")
        result = Channel(ctx, key, normalized)
        ctx.channels[key] = result
        return result


def _validate_manifest(ctx, manifest, ch, objects=None):
    if manifest.get("schema_version") != 1 or manifest.get("profile") != ctx.profile:
        raise ValueError("kernel distributed ABI/profile does not match the session")
    groups = manifest.get("groups", [])
    if len(groups) != 1 or groups[0].get("size") != ctx.world_size:
        raise ValueError("kernel group size does not match the session")
    channels = manifest.get("channels", [])
    if len(channels) != 1 or channels[0].get("name") != ch.key or channels[0].get("group") != groups[0]["name"]:
        raise ValueError("kernel channel binding does not match the prepared channel")
    specs = manifest.get("objects", [])
    if not specs or len({obj["key"] for obj in specs}) != len(specs):
        raise ValueError("kernel manifest must contain unique symmetric object keys")
    bound = ctx.objects if objects is None else objects
    expected_keys = [obj["key"] for obj in specs]
    if [key for key in bound if key in expected_keys] != expected_keys:
        raise ValueError("symmetric allocation order does not match the kernel manifest")
    if objects is not None and set(bound) != set(expected_keys):
        raise ValueError("symmetric bindings must match exactly the kernel manifest")
    for spec in specs:
        obj = bound.get(spec["key"])
        if obj is None:
            raise ValueError(f"missing symmetric allocation {spec['key']!r}")
        expected = {
            "shape": list(obj.shape),
            "dtype": obj.dtype,
            "kind": obj.kind,
            "remote_access": obj.remote_access,
            "strides": list(obj.strides),
            "physical_nbytes": obj.physical_nbytes,
            "alignment": 64,
            "group": groups[0]["name"],
        }
        for key, value in expected.items():
            if spec.get(key) != value:
                raise ValueError(f"symmetric object {obj.key!r} has mismatched {key}: manifest={spec.get(key)!r}, allocation={value!r}")


def prepare_launch(ctx, kernel, *, channel, epochs, protocol, parameters=None, binding=None):
    if ctx.profile == NATIVE_PROFILE:
        return ctx.prepare(kernel, channel=channel, epochs=epochs, protocol=protocol, parameters=parameters, binding=binding)
    if binding is not None:
        raise UnsupportedProfileError("program bindings currently require a native session")
    with ctx._lock:
        if ctx.state == "LOCAL_DONE":
            # One synchronous coordinator has already completed every logical rank.
            ctx.state = "QUIESCENT"
        if ctx.state != "QUIESCENT":
            raise RuntimeError(f"prepare_launch requires a QUIESCENT session; found {ctx.state}")
        if channel.session is not ctx or ctx.channels.get(channel.key) is not channel:
            raise ValueError("channel belongs to a different session")
        epochs = _positive_int(epochs, "epochs")
        if ctx._epoch + epochs > MAX_EPOCH:
            raise OverflowError("int32 signal epoch space exhausted; close and recreate the session")
        if not isinstance(protocol, str) or not protocol:
            raise ValueError("protocol must be a nonempty stable name")
        parameters = canonical_manifest({} if parameters is None else parameters)
        manifest = kernel_manifest(kernel)
        _validate_manifest(ctx, manifest, channel)
        launch_id, end = ctx._launch_id + 1, ctx._epoch + epochs
        descriptors = {
            rank: encode_descriptor(
                manifest, rank=rank, generation=ctx.generation, launch_id=launch_id, epoch_base=ctx._epoch, epoch_end=end
            )
            for rank in range(ctx.world_size)
        }
        ticket = LaunchTicket(
            ctx,
            kernel,
            channel,
            ctx.generation,
            launch_id,
            ctx._epoch,
            end,
            protocol,
            parameters,
            manifest,
            manifest_digest(manifest),
            descriptors,
        )
        ctx._ticket, ctx._epoch, ctx._launch_id, ctx.state = ticket, end, launch_id, "PREPARED"
        return ticket


def _check_ticket(ctx, ticket):
    if ticket.session is not ctx or ctx._ticket is not ticket or ticket.generation != ctx.generation:
        raise ValueError("ticket is stale or belongs to a different session")


def copy_in(buffer, value, *, ticket, rank=None, stream=None):
    ctx = buffer.session
    if ctx.profile == NATIVE_PROFILE:
        return ctx.copy_in(buffer, value, ticket=ticket, rank=rank, stream=stream)
    with ctx._lock:
        _check_ticket(ctx, ticket)
        if ctx.state != "PREPARED":
            raise RuntimeError("copy_in requires the current PREPARED ticket")
        if stream is not None:
            raise UnsupportedProfileError("CPU co-simulation does not accept a device stream")
        if buffer.kind != "symmetric":
            raise ValueError("signals are initialized only at allocation and cannot be overwritten by copy_in")
        if not any(spec["key"] == buffer.key for spec in ticket.manifest["objects"]):
            raise ValueError("copy_in buffer is not bound by this kernel manifest")
        if isinstance(value, dict):
            if rank is not None or set(value) != set(range(ctx.world_size)):
                raise ValueError("rank input mapping must cover every rank and omit rank=")
            items = value.items()
        else:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank not in buffer.arrays:
                raise ValueError("copy_in of one array requires an in-group rank=")
            items = [(rank, value)]
        copies = []
        for owner, array in items:
            array = np.asarray(array)
            dst = buffer.arrays[owner]
            if array.shape != dst.shape or array.dtype != dst.dtype:
                raise ValueError(f"copy_in {buffer.key} rank {owner} expects shape {dst.shape} and storage dtype {dst.dtype}")
            copies.append((dst, array))
        for dst, array in copies:
            np.copyto(dst, array)


def close(ctx):
    if ctx.profile == NATIVE_PROFILE:
        return ctx.close()
    with ctx._lock:
        if ctx.state == "CLOSED":
            return
        if ctx.state in ("RUNNING", "FAILED"):
            raise RuntimeError(f"cannot close session in {ctx.state}; preserve its allocations and failure evidence")
        ctx.state = "CLOSED"
        ctx._ticket = None
