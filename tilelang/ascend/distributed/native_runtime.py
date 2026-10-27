"""Host ownership and bootstrap for the explicit two-process SHMEM profile.

Payloads remain in the symmetric NPU heap. TCPStore exchanges JSON metadata
only. Importing this module does not load SHMEM, Torch NPU, or initialize ACL.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from datetime import timedelta
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile
from threading import RLock
from types import MappingProxyType
from urllib.parse import urlparse

import numpy as np

from tilelang.distributed.abi import (
    MAX_EPOCH,
    NATIVE_PROFILE,
    NATIVE_PROFILES,
    SIGNAL_STRIDE_BYTES,
    canonical_manifest,
    kernel_manifest,
    manifest_digest,
)
from tilelang.distributed.session import Channel, _contiguous_strides, _positive_int, _storage_dtype

SHMEM_REVISION = "a232a76516c435cd39ecd10125fc6e14a55fc18b"
_FAILED_SESSIONS = []  # Keep in-flight allocations alive until the worker exits.
_LOADED_BRIDGES = []  # DLPack deleters must outlive their shared-library code.
_NATIVE_SESSION_LOCK = RLock()
_ACTIVE_NATIVE_SESSION = None


def _locked(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return invoke


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _allocation_layout(*, shape, dtype, remote_access, kind):
    shape = tuple(_positive_int(x, "shape extent") for x in shape)
    if not shape or kind not in ("symmetric", "signal") or remote_access not in ("read", "write", "read_write"):
        raise ValueError("invalid native symmetric allocation specification")
    dtype = str(dtype)
    itemsize = _storage_dtype(dtype).itemsize
    if kind == "signal":
        if dtype != "int32":
            raise ValueError("native SignalValue must be int32")
        remote_access = "write"
    nbytes = math.prod(shape) * (SIGNAL_STRIDE_BYTES if kind == "signal" else itemsize)
    strides = tuple(s * (SIGNAL_STRIDE_BYTES // itemsize if kind == "signal" else 1) for s in _contiguous_strides(shape))
    return shape, dtype, remote_access, nbytes, strides


def _loopback_bootstrap(value):
    parsed = urlparse(value or "")
    if parsed.scheme != "tcp" or parsed.hostname != "127.0.0.1" or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("bootstrap must be tcp://127.0.0.1:<port> for this single-host profile")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("bootstrap requires a valid TCP port") from error
    if port is None or not 1 <= port <= 65535 or parsed.username or parsed.password:
        raise ValueError("bootstrap requires a valid TCP port")
    return port


class StoreControl:
    """Bounded metadata collectives without an HCCL process group."""

    def __init__(self, *, rank, world_size, port, timeout):
        import torch.distributed as dist

        self.rank, self.world_size, self.sequence = rank, world_size, 0
        self.store = dist.TCPStore("127.0.0.1", port, world_size, rank == 0, timedelta(seconds=timeout))
        self.timeout = timedelta(seconds=timeout)

    def all_gather(self, value):
        encoded = _json(value)
        if len(encoded.encode()) > 1024 * 1024:
            raise ValueError("control metadata exceeds 1 MiB; tensor payloads cannot use this channel")
        prefix = f"p0/{self.sequence}"
        self.sequence += 1
        keys = [f"{prefix}/{rank}" for rank in range(self.world_size)]
        self.store.set(keys[self.rank], encoded)
        self.store.wait(keys, self.timeout)
        return [json.loads(self.store.get(key)) for key in keys]


def build_host_bridge(*, shmem_root, shmem_library=None, cann_root=None, output_dir=None):
    """Build the small adapter using real SDK headers, without touching a device."""
    root = Path(shmem_root).expanduser().resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != SHMEM_REVISION:
        raise ValueError(f"native SHMEM requires source revision {SHMEM_REVISION}; found {revision}")
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "include", "src"], check=True, capture_output=True)
    library = Path(shmem_library).expanduser().resolve() if shmem_library else root / "install/shmem/lib/libshmem.so"
    if not library.is_file():
        raise FileNotFoundError(f"SHMEM library does not exist: {library}; build the pinned source for Ascend950 first")
    cann = Path(cann_root or os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")).resolve()
    source = Path(__file__).resolve().parents[2] / "src/ascend/distributed/shmem_host_bridge.cc"
    if not source.is_file():
        source = Path(__file__).resolve().parents[3] / "src/ascend/distributed/shmem_host_bridge.cc"
    import tvm_ffi

    dlpack = Path(tvm_ffi.__file__).parent / "include"
    if not (dlpack / "dlpack/dlpack.h").is_file():
        dlpack = Path(__file__).resolve().parents[3] / "3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise FileNotFoundError("a host C++ compiler is required to build the SHMEM adapter")
    compiler_version = subprocess.check_output([compiler, "--version"], text=True, timeout=30)
    headers = hashlib.sha256()
    for header in sorted((cann / "include/acl").rglob("*.h")):
        headers.update(str(header.relative_to(cann)).encode())
        headers.update(header.read_bytes())
    headers.update((dlpack / "dlpack/dlpack.h").read_bytes())
    headers_hash = headers.hexdigest()
    identity = hashlib.sha256(
        _json([source_hash, library_hash, str(cann), revision, compiler, compiler_version, headers_hash]).encode()
    ).hexdigest()
    directory = (Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "tilelang-p0-shmem") / identity
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "libtilelang_shmem_host.so"
    # A process-specific temporary avoids two ranks observing a partially linked file.
    if not output.is_file():
        temporary = directory / f"bridge-{os.getpid()}.so"
        command = [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            str(source),
            f"-I{root / 'include'}",
            f"-I{cann / 'include'}",
            f"-I{dlpack}",
            str(library),
            f"-L{cann / 'lib64'}",
            "-lascendcl",
            f"-Wl,-rpath,{library.parent}",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-o",
            str(temporary),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        temporary.replace(output)
    return output, {
        "shmem_revision": revision,
        "shmem_library_sha256": library_hash,
        "bridge_source_sha256": source_hash,
        "sdk_headers_sha256": headers_hash,
        "host_compiler": compiler_version,
        "bridge_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


class ShmemBridge:
    def __init__(self, *, shmem_root, shmem_library=None):
        path, self.fingerprint = build_host_bridge(shmem_root=shmem_root, shmem_library=shmem_library)
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        _LOADED_BRIDGES.append(self.lib)
        u64, i32, ptr = ctypes.c_uint64, ctypes.c_int32, ctypes.c_void_p
        signatures = {
            "init": (i32, [i32, i32, u64, ctypes.c_char_p, ctypes.c_uint32]),
            "finalize": (i32, []),
            "alloc": (u64, [u64]),
            "free": (None, [u64]),
            "peer": (u64, [u64, i32]),
            "zero": (i32, [u64, u64]),
            "h2d": (i32, [u64, ptr, u64]),
            "d2h": (i32, [ptr, u64, u64]),
            "stream_create": (i32, [ctypes.POINTER(ptr)]),
            "stream_sync": (i32, [ptr, i32]),
            "stream_destroy": (i32, [ptr]),
            "set_device": (i32, [i32]),
            "physical_device": (i32, [i32, ctypes.POINTER(i32)]),
            "soc": (ctypes.c_char_p, []),
            "error": (ctypes.c_char_p, []),
            "live_views": (u64, []),
            "tensor": (
                u64,
                [u64, i32, ctypes.c_uint8, ctypes.c_uint8, i32, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)],
            ),
            "delete_tensor": (None, [u64]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(self.lib, f"tl_shmem_{name}")
            function.restype, function.argtypes = restype, argtypes

    def check(self, name, *args):
        result = getattr(self.lib, f"tl_shmem_{name}")(*args)
        if result != 0:
            detail = self.lib.tl_shmem_error()
            raise RuntimeError(f"SHMEM/ACL {name} failed ({result}): {detail.decode(errors='replace') if detail else ''}")

    def device_identity(self, device):
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(device)
        soc = self.lib.tl_shmem_soc()
        if not soc or not soc.decode().startswith("Ascend950"):
            raise RuntimeError(f"native profile requires Ascend950; runtime reports {soc!r}")
        physical = ctypes.c_int32()
        self.check("physical_device", device, ctypes.byref(physical))
        return {"physical_device": physical.value, "soc": soc.decode()}

    def initialize(self, *, rank, world_size, device, heap_bytes, bootstrap, timeout):
        self.check("set_device", device)
        self.check("init", rank, world_size, heap_bytes, bootstrap.encode(), timeout)
        stream = ctypes.c_void_p()
        self.check("stream_create", ctypes.byref(stream))
        return stream.value

    def activate(self, device):
        self.check("set_device", device)

    def allocate(self, nbytes):
        pointer = self.lib.tl_shmem_alloc(nbytes)
        if not pointer or pointer % 64:
            raise RuntimeError("SHMEM aligned allocation failed or returned a non-aligned pointer")
        self.check("zero", pointer, nbytes)
        return pointer

    def peer(self, pointer, rank):
        result = self.lib.tl_shmem_peer(pointer, rank)
        if not result:
            raise RuntimeError(f"SHMEM MTE mapping unavailable for peer rank {rank}")
        return result

    def copy_in(self, pointer, array):
        self.check("h2d", pointer, array.ctypes.data, array.nbytes)

    def copy_out(self, pointer, array):
        self.check("d2h", array.ctypes.data, pointer, array.nbytes)

    def tensor(self, *, pointer, device, shape, dtype, strides):
        import tvm_ffi

        dt = tvm_ffi.dtype(dtype)
        axes = ctypes.c_int64 * len(shape)
        managed = self.lib.tl_shmem_tensor(pointer, device, dt.type_code, dt.bits, len(shape), axes(*shape), axes(*strides))
        if not managed:
            raise MemoryError("could not create a borrowed symmetric tensor view")
        create_capsule = ctypes.pythonapi.PyCapsule_New
        create_capsule.restype = ctypes.py_object
        create_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        capsule = create_capsule(managed, b"dltensor", None)
        try:
            return tvm_ffi.from_dlpack(capsule)
        except Exception:
            is_valid = ctypes.pythonapi.PyCapsule_IsValid
            is_valid.restype = ctypes.c_int
            is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
            if is_valid(capsule, b"dltensor"):
                self.lib.tl_shmem_delete_tensor(managed)
            raise

    def synchronize(self, stream, timeout):
        self.check("stream_sync", stream, timeout * 1000)

    def launch(self, kernel, args, *, device, stream):
        import tvm_ffi

        self.check("set_device", device)
        with tvm_ffi.use_raw_stream(tvm_ffi.Device(12, device), stream):
            return kernel(*args)

    def live_views(self):
        return self.lib.tl_shmem_live_views()

    def close(self, pointers, stream):
        self.check("stream_destroy", stream)
        for pointer in reversed(pointers):
            self.lib.tl_shmem_free(pointer)
        self.check("finalize")


@dataclass(frozen=True, eq=False)
class NativeBuffer:
    session: NativeSession
    key: str
    shape: tuple
    dtype: str
    remote_access: str
    kind: str
    pointer: int
    physical_nbytes: int
    strides: tuple

    @property
    def _lock(self):
        return self.session._lock

    @_locked
    def view(self, shape=None, dtype=None, byte_offset=0):
        if self.session.objects.get(self.key) is not self:
            raise ValueError("buffer is no longer registered with its owning session")
        if self.session.state in ("CLOSED", "FAILED"):
            raise RuntimeError("cannot create a view from a closed or failed session")
        dtype = dtype or self.dtype
        shape = self.shape if shape is None else tuple(_positive_int(x, "view shape") for x in shape)
        itemsize = _storage_dtype(dtype).itemsize
        if isinstance(byte_offset, bool) or not isinstance(byte_offset, int) or byte_offset < 0 or byte_offset % itemsize:
            raise ValueError("view byte_offset must be nonnegative and dtype aligned")
        strides = self.strides if shape == self.shape and dtype == self.dtype and byte_offset == 0 else _contiguous_strides(shape)
        nbytes = (1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))) * itemsize
        if not shape or byte_offset + nbytes > self.physical_nbytes:
            raise ValueError("tensor view exceeds the symmetric allocation")
        return self.session.bridge.tensor(
            pointer=self.pointer + byte_offset, device=self.session.device, shape=shape, dtype=dtype, strides=strides
        )

    @property
    def local_tensor(self):
        return self.view()

    def for_rank(self, rank):
        if rank != self.session.rank:
            raise ValueError("native buffers expose only this process's local rank")
        return self.view()


@dataclass(frozen=True, eq=False)
class NativeBinding:
    """One program's persistent allocations in the Session's fixed SHMEM heap."""

    session: NativeSession
    kernel: object
    key: str
    objects: MappingProxyType
    channel: Channel
    manifest_digest: str
    signal_values: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class NativeTicket:
    session: NativeSession
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
    binding: NativeBinding | None = None

    @property
    def epochs(self):
        return self.end - self.base


class NativeSession:
    def __init__(
        self,
        *,
        rank,
        world_size,
        device,
        profile=NATIVE_PROFILE,
        bootstrap,
        control,
        bridge,
        heap_bytes=64 * 1024 * 1024,
        timeout=60,
        shmem_root=None,
        shmem_library=None,
    ):
        self._lock = RLock()
        self.rank, self.world_size, self.device = rank, world_size, device
        self.profile, self.state = profile, "INITIALIZING"
        self.control, self.bridge, self.timeout = control, bridge, timeout
        self.shmem_root = None if shmem_root is None else Path(shmem_root).expanduser().resolve()
        self.shmem_library = shmem_library
        self.abi_fingerprint = bridge.fingerprint
        self.objects, self.channels = {}, {}
        self._bindings = {}
        self._epoch = self._launch_id = 0
        self._ticket = None
        self._allocated_bytes = 0
        self.heap_bytes = heap_bytes
        self.error = None
        self.stream = None
        if isinstance(bridge, ShmemBridge):
            global _ACTIVE_NATIVE_SESSION
            with _NATIVE_SESSION_LOCK:
                if _ACTIVE_NATIVE_SESSION is not None:
                    raise RuntimeError("only one native SHMEM session may be alive per process; close it before initializing another")
                _ACTIVE_NATIVE_SESSION = self
        try:
            device_info = (
                bridge.device_identity(device) if hasattr(bridge, "device_identity") else {"physical_device": device, "soc": "fake"}
            )
        except Exception as error:
            self.fail(error)
            raise
        self.physical_device = device_info["physical_device"]
        info = {
            **device_info,
            "rank": rank,
            "device": device,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "nonce": secrets.randbits(63) + 1,
            "profile": self.profile,
            "abi": self.abi_fingerprint,
            "bootstrap": bootstrap,
            "heap_bytes": heap_bytes,
            "timeout": timeout,
        }
        try:
            peers = self.control.all_gather(info)
            if len(peers) != 2 or [x["rank"] for x in peers] != [0, 1]:
                raise ValueError("native bootstrap requires exactly ranks 0 and 1")
            if len({x["host"] for x in peers}) != 1 or len({x["physical_device"] for x in peers}) != 2:
                raise ValueError("native ranks must select different devices on one host")
            for key in ("profile", "abi", "bootstrap", "heap_bytes", "timeout", "soc"):
                if peers[0][key] != peers[1][key]:
                    raise ValueError(f"native bootstrap metadata mismatch: {key}")
            self.generation = peers[0]["nonce"]
            self.stream = bridge.initialize(
                rank=rank, world_size=world_size, device=device, heap_bytes=heap_bytes, bootstrap=bootstrap, timeout=timeout
            )
            self.preflight({"initialized": True})
            self.state = "QUIESCENT"
        except Exception as error:
            self.fail(error)
            raise

    def _activate(self):
        if hasattr(self.bridge, "activate"):
            self.bridge.activate(self.device)

    def __enter__(self):
        if self.state in ("CLOSED", "FAILED"):
            raise RuntimeError(f"cannot enter a native Session in {self.state}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # An escaping rank error must not enter a cleanup collective or
            # free storage that another rank might still access.
            self.fail(exc_value)
        elif self.state != "FAILED":
            self.close()
        return False

    @_locked
    def preflight(self, payload):
        try:
            payload = canonical_manifest(payload)
            gathered = self.control.all_gather(payload)
            if len(gathered) != self.world_size or any(item != payload for item in gathered):
                failures = [
                    f"rank {rank}: {str(item['error'])[:2048]}"
                    for rank, item in enumerate(gathered)
                    if isinstance(item, dict) and item.get("error")
                ]
                details = "; " + "; ".join(failures) if failures else ""
                raise ValueError("native cross-rank metadata mismatch; no kernel may be launched" + details)
        except Exception as error:
            self.fail(error)
            raise

    @_locked
    def fail(self, error):
        self.error, self.state = str(error), "FAILED"
        if self not in _FAILED_SESSIONS:
            _FAILED_SESSIONS.append(self)

    @_locked
    def allocate(self, *, key, shape, dtype, remote_access="read_write", kind="symmetric"):
        if self.state != "QUIESCENT":
            raise RuntimeError("symmetric allocation requires a QUIESCENT session")
        if not isinstance(key, str) or not key or key in self.objects:
            raise ValueError("symmetric object keys must be nonempty and unique")
        shape, dtype, remote_access, nbytes, strides = _allocation_layout(shape=shape, dtype=dtype, remote_access=remote_access, kind=kind)
        if self._allocated_bytes + nbytes > self.heap_bytes:
            raise MemoryError("native symmetric allocations exceed the configured heap capacity")
        self.preflight(
            {
                "allocate": key,
                "shape": shape,
                "dtype": dtype,
                "kind": kind,
                "remote_access": remote_access,
                "nbytes": nbytes,
                "index": len(self.objects),
            }
        )
        try:
            self._activate()
            pointer = self.bridge.allocate(nbytes)
            # Retain ownership immediately, including on mapping/preflight failure.
            result = NativeBuffer(self, key, shape, dtype, remote_access, kind, pointer, nbytes, strides)
            self.objects[key] = result
            self._allocated_bytes += nbytes
            self.bridge.peer(pointer, 1 - self.rank)
            self.preflight({"allocated": key})
            return result
        except Exception as error:
            self.fail(error)
            raise

    @_locked
    def bind(self, kernel, *, key=None):
        """Collectively cache isolated program resources without resizing the heap.

        Every rank must bind programs in the same order, including cache hits.
        Allocations remain owned by the Session until it closes.
        """
        from tilelang.distributed.session import _validate_manifest

        if self.state not in ("QUIESCENT", "LOCAL_DONE"):
            raise RuntimeError(f"bind requires a quiescent or completed session; found {self.state}")
        manifest = kernel_manifest(kernel)
        digest = manifest_digest(manifest)
        key = digest if key is None else key
        if not isinstance(key, str) or not key:
            raise ValueError("binding key must be a nonempty string")
        cached = self._bindings.get(key)
        # Checking cache state collectively prevents one rank returning while
        # another rank enters symmetric allocation collectives.
        self.preflight(
            {
                "bind": key,
                "manifest": digest,
                "cached": cached is not None,
                "allocated_bytes": self._allocated_bytes,
                "heap_bytes": self.heap_bytes,
                "launch_id": self._launch_id,
            }
        )
        if cached is not None:
            if cached.manifest_digest != digest:
                raise ValueError("binding key is already associated with a different program manifest")
            self._check_binding(cached)
            return cached

        channels = manifest.get("channels", [])
        if len(channels) != 1 or not isinstance(channels[0].get("name"), str) or not channels[0]["name"]:
            raise ValueError("native bindings require one named channel")
        channel_key = channels[0]["name"]
        peers = {0: (1,), 1: (0,)}
        channel = self.channels.get(channel_key, Channel(self, channel_key, peers))
        if channel.session is not self or channel.peers != peers:
            raise ValueError("native binding channel must contain the complete two-rank topology")
        namespace = hashlib.sha256(key.encode()).hexdigest()
        layouts = {}
        for spec in manifest.get("objects", []):
            logical_key = spec["key"]
            if not isinstance(logical_key, str) or not logical_key or logical_key in layouts:
                raise ValueError("binding manifest requires unique nonempty object keys")
            physical_key = f"__binding__/{namespace}/{logical_key}"
            if physical_key in self.objects:
                raise ValueError("binding allocation namespace conflicts with an existing allocation")
            shape, dtype, remote_access, nbytes, strides = _allocation_layout(
                shape=spec["shape"], dtype=spec["dtype"], remote_access=spec["remote_access"], kind=spec["kind"]
            )
            # These address-free layout records validate the entire manifest
            # and heap budget before the first device allocation.
            layouts[logical_key] = NativeBuffer(self, physical_key, shape, dtype, remote_access, spec["kind"], 0, nbytes, strides)
        _validate_manifest(self, manifest, channel, objects=layouts)
        required = sum(obj.physical_nbytes for obj in layouts.values())
        self.preflight({"bind_capacity": key, "required_bytes": required, "available_bytes": self.heap_bytes - self._allocated_bytes})
        if self._allocated_bytes + required > self.heap_bytes:
            raise MemoryError(
                f"binding requires {required} bytes but the fixed SHMEM heap has "
                f"{self.heap_bytes - self._allocated_bytes} bytes available; recreate the Session with a larger heap_bytes"
            )
        # LOCAL_DONE includes the previous launch's collective completion.
        self.state = "QUIESCENT"
        objects = {
            logical_key: self.allocate(key=obj.key, shape=obj.shape, dtype=obj.dtype, remote_access=obj.remote_access, kind=obj.kind)
            for logical_key, obj in layouts.items()
        }
        self.channels[channel_key] = channel
        signal_values = {
            (rank, logical_key, index): 0
            for logical_key, obj in objects.items()
            if obj.kind == "signal"
            for rank in range(self.world_size)
            for index in range(math.prod(obj.shape))
        }
        binding = NativeBinding(self, kernel, key, MappingProxyType(objects), channel, digest, signal_values)
        self._bindings[key] = binding
        return binding

    def _check_binding(self, binding):
        from tilelang.distributed.session import _validate_manifest

        if not isinstance(binding, NativeBinding) or binding.session is not self or self._bindings.get(binding.key) is not binding:
            raise ValueError("binding is stale or belongs to another session")
        manifest = kernel_manifest(binding.kernel)
        if manifest_digest(manifest) != binding.manifest_digest:
            raise ValueError("bound program manifest was modified")
        if self.channels.get(binding.channel.key) is not binding.channel:
            raise ValueError("binding channel is no longer registered with its Session")
        for obj in binding.objects.values():
            if obj.session is not self or self.objects.get(obj.key) is not obj:
                raise ValueError("bound allocation is no longer owned by its Session")
        _validate_manifest(self, manifest, binding.channel, objects=binding.objects)

    @_locked
    def channel(self, *, key, peers=None):
        if self.state != "QUIESCENT" or self.channels:
            raise RuntimeError("P0 channel creation requires a QUIESCENT session with no existing channel")
        expected = {0: (1,), 1: (0,)}
        normalized = expected if peers is None else {rank: tuple(values) for rank, values in peers.items()}
        if normalized != expected or not isinstance(key, str) or not key:
            raise ValueError("native P0 requires the complete two-rank peer channel")
        self.preflight({"channel": key, "peers": {str(rank): values for rank, values in normalized.items()}})
        result = Channel(self, key, normalized)
        self.channels[key] = result
        return result

    @_locked
    def prepare(self, kernel, *, channel, epochs, protocol, parameters=None, binding=None):
        if self.state == "LOCAL_DONE":
            self.state = "QUIESCENT"
        if self.state != "QUIESCENT":
            raise RuntimeError(f"prepare requires a QUIESCENT session; found {self.state}")
        if channel.session is not self or self.channels.get(channel.key) is not channel:
            raise ValueError("channel belongs to another session")
        epochs = _positive_int(epochs, "epochs")
        if self._epoch + epochs > MAX_EPOCH:
            raise OverflowError("int32 signal epochs exhausted; close and recreate the session")
        if not isinstance(protocol, str) or not protocol:
            raise ValueError("protocol must be a nonempty stable name")
        manifest = kernel_manifest(kernel)
        if manifest.get("profile") != self.profile:
            raise ValueError("native kernel profile does not match the session")
        # Reuse the shared manifest binding contract; it is independent of storage ownership.
        from tilelang.distributed.session import _validate_manifest

        if binding is not None:
            self._check_binding(binding)
            if kernel is not binding.kernel or channel is not binding.channel:
                raise ValueError("prepare must use the bound program and channel")
        _validate_manifest(self, manifest, channel, objects=None if binding is None else binding.objects)
        parameters = canonical_manifest({} if parameters is None else parameters)
        launch_id, end = self._launch_id + 1, self._epoch + epochs
        prim_func = getattr(kernel, "prim_func", None)
        needs_launch_scalars = prim_func is not None and any(
            parameter not in prim_func.buffer_map and parameter.name != "epoch0" for parameter in prim_func.params
        )
        # User scalars are bound by D.launch(scalars=...). The complete plan is
        # always validated there before compilation or any device submission.
        if hasattr(kernel, "plan") and not needs_launch_scalars:
            kernel.plan(epoch_base=self._epoch, epoch_end=end)
        digest = manifest_digest(manifest)
        self.preflight(
            {
                "prepare": launch_id,
                "generation": self.generation,
                "base": self._epoch,
                "end": end,
                "protocol": protocol,
                "parameters": parameters,
                "manifest": digest,
                "binding": None if binding is None else binding.key,
            }
        )
        ticket = NativeTicket(
            self, kernel, channel, self.generation, launch_id, self._epoch, end, protocol, parameters, manifest, digest, binding
        )
        self._ticket, self._epoch, self._launch_id, self.state = ticket, end, launch_id, "PREPARED"
        return ticket

    def check_ticket(self, ticket):
        if ticket.session is not self or self._ticket is not ticket or ticket.generation != self.generation:
            raise ValueError("ticket is stale or belongs to another session")

    @_locked
    def ticket_buffers(self, ticket):
        """Resolve logical argument names to this ticket's Session-owned buffers."""
        self.check_ticket(ticket)
        if manifest_digest(ticket.manifest) != ticket.manifest_digest:
            raise ValueError("prepared manifest was modified")
        if manifest_digest(kernel_manifest(ticket.kernel)) != ticket.manifest_digest:
            raise ValueError("prepared program manifest was modified")
        if ticket.binding is not None:
            self._check_binding(ticket.binding)
            if ticket.kernel is not ticket.binding.kernel or ticket.channel is not ticket.binding.channel:
                raise ValueError("ticket does not match its program binding")
            return dict(ticket.binding.objects)
        return {spec["key"]: self.objects[spec["key"]] for spec in ticket.manifest["objects"]}

    @_locked
    def copy_in(self, buffer, value, *, ticket, rank=None, stream=None):
        self.check_ticket(ticket)
        if self.state != "PREPARED" or buffer.session is not self or self.objects.get(buffer.key) is not buffer:
            raise RuntimeError("copy_in requires this session's buffer and current PREPARED ticket")
        if stream is not None or isinstance(rank, bool) or (rank is not None and rank != self.rank) or isinstance(value, dict):
            raise ValueError("native copy_in accepts only the local rank's array and the session-owned stream")
        if buffer.kind == "signal":
            raise ValueError("signals cannot be overwritten by copy_in")
        if not any(buffer is obj for obj in self.ticket_buffers(ticket).values()):
            raise ValueError("copy_in buffer is not bound by the kernel manifest")
        array = np.asarray(value)
        if array.shape != buffer.shape or array.dtype != _storage_dtype(buffer.dtype):
            raise ValueError(f"copy_in {buffer.key} expects {buffer.shape} / {_storage_dtype(buffer.dtype)}")
        try:
            self._activate()
            self.bridge.copy_in(buffer.pointer, np.ascontiguousarray(array))
        except Exception as error:
            self.fail(error)
            raise

    @_locked
    def copy_out(self, buffer):
        if self.state not in ("QUIESCENT", "PREPARED", "LOCAL_DONE") or buffer.session is not self:
            raise RuntimeError("copy_out requires a locally completed or unlaunched session")
        try:
            self._activate()
            if buffer.kind == "signal":
                raw = np.empty(buffer.physical_nbytes // 4, dtype=np.int32)
                self.bridge.copy_out(buffer.pointer, raw)
                return raw[::16].reshape(buffer.shape).copy()
            result = np.empty(buffer.shape, dtype=_storage_dtype(buffer.dtype))
            self.bridge.copy_out(buffer.pointer, result)
            return result
        except Exception as error:
            self.fail(error)
            raise

    @_locked
    def mark_running(self, ticket):
        self.check_ticket(ticket)
        if self.state != "PREPARED":
            raise RuntimeError("launch requires the current PREPARED ticket")
        self.preflight({"launch": ticket.launch_id, "generation": ticket.generation})
        self.state = "RUNNING"

    @_locked
    def launch_kernel(self, kernel, *args):
        if self.state != "RUNNING":
            raise RuntimeError("native kernels may only launch in RUNNING state")
        try:
            return self.bridge.launch(kernel, args, device=self.device, stream=self.stream)
        except Exception as error:
            self.fail(error)
            raise

    @_locked
    def synchronize(self):
        try:
            self._activate()
            self.bridge.synchronize(self.stream, self.timeout)
        except Exception as error:
            self.fail(error)
            raise

    @_locked
    def mark_complete(self, ticket):
        self.check_ticket(ticket)
        if self.state != "RUNNING":
            raise RuntimeError("completion requires RUNNING state")
        self.synchronize()
        self.preflight({"completed": ticket.launch_id, "end": ticket.end, "generation": ticket.generation})
        self.state = "LOCAL_DONE"

    @_locked
    def close(self):
        if self.state == "CLOSED":
            return
        if self.state not in ("QUIESCENT", "PREPARED", "LOCAL_DONE"):
            raise RuntimeError(f"cannot close native session in {self.state}; terminate the worker to preserve in-flight allocations")
        if self.bridge.live_views():
            raise RuntimeError("release all borrowed symmetric tensor views before closing the session")
        self.synchronize()
        self.preflight({"close": self.generation, "launch_id": self._launch_id})
        try:
            self.bridge.close([obj.pointer for obj in self.objects.values()], self.stream)
            self.state, self._ticket = "CLOSED", None
            self._bindings.clear()
            global _ACTIVE_NATIVE_SESSION
            with _NATIVE_SESSION_LOCK:
                if _ACTIVE_NATIVE_SESSION is self:
                    _ACTIVE_NATIVE_SESSION = None
        except Exception as error:
            self.fail(error)
            raise


def init_runtime(
    *,
    profile=NATIVE_PROFILE,
    process_group=None,
    device=None,
    world_size=2,
    rank=None,
    bootstrap=None,
    control_port=None,
    shmem_root=None,
    shmem_library=None,
    heap_bytes=64 * 1024 * 1024,
    timeout=60,
    control=None,
    bridge=None,
):
    if profile not in NATIVE_PROFILES or process_group is not None or world_size != 2:
        raise ValueError("native profile requires exactly two ranks and its own metadata control channel")
    rank = int(os.environ.get("RANK", "0")) if rank is None else rank
    device = int(os.environ.get("LOCAL_RANK", str(rank))) if device is None else device
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in (0, 1):
        raise ValueError("rank must be 0 or 1")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("device must be a nonnegative process-visible NPU index")
    timeout, heap_bytes = _positive_int(timeout, "timeout"), _positive_int(heap_bytes, "heap_bytes")
    if timeout * 1000 >= 1 << 31:
        raise ValueError("timeout exceeds ACL's int32 millisecond range")
    bootstrap_port = _loopback_bootstrap(bootstrap)
    if control is None:
        if (
            isinstance(control_port, bool)
            or not isinstance(control_port, int)
            or not 1 <= control_port <= 65535
            or control_port == bootstrap_port
        ):
            raise ValueError("control_port must be a valid port distinct from the SHMEM bootstrap port")
        control = StoreControl(rank=rank, world_size=world_size, port=control_port, timeout=timeout)
    shmem_root = shmem_root or os.environ.get("SHMEM_ROOT")
    if bridge is None:
        if not shmem_root:
            raise ValueError("set SHMEM_ROOT or pass shmem_root pointing to the pinned SHMEM source")
        bridge = ShmemBridge(shmem_root=shmem_root, shmem_library=shmem_library)
    return NativeSession(
        profile=profile,
        rank=rank,
        world_size=world_size,
        device=device,
        bootstrap=bootstrap,
        control=control,
        bridge=bridge,
        heap_bytes=heap_bytes,
        timeout=timeout,
        shmem_root=shmem_root,
        shmem_library=shmem_library,
    )
