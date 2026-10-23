"""Hardware-free ownership and bootstrap tests for the native SHMEM runtime."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tilelang.ascend.distributed.native_runtime import NATIVE_PROFILE, NativeSession, init_runtime, _loopback_bootstrap
from tilelang.distributed.abi import MAX_EPOCH


class Control:
    def __init__(self):
        self.payloads = []
        self.mismatch = False

    def all_gather(self, payload):
        self.payloads.append(payload)
        peer = copy.deepcopy(payload)
        if "nonce" in peer:
            peer["rank"], peer["device"] = 1 - payload["rank"], 1 - payload["device"]
            peer["physical_device"] = 1 - payload["physical_device"]
            return [payload, peer] if payload["rank"] == 0 else [peer, payload]
        if self.mismatch:
            peer["mismatch"] = True
        return [payload, peer]


class Bridge:
    fingerprint = {"test": "deterministic fake"}

    def __init__(self):
        self.calls = []
        self.memory = {}
        self.next_pointer = 64
        self.fail_sync = False
        self.views = 0

    def initialize(self, **kwargs):
        self.calls.append(("initialize", kwargs))
        return 1234

    def activate(self, device):
        self.calls.append(("activate", device))

    def allocate(self, nbytes):
        self.calls.append(("allocate", nbytes))
        pointer = self.next_pointer
        self.next_pointer += ((nbytes + 63) // 64) * 64
        self.memory[pointer] = np.zeros(nbytes, dtype=np.uint8)
        return pointer

    def peer(self, pointer, rank):
        self.calls.append(("peer", pointer, rank))
        return pointer + 65536

    def copy_in(self, pointer, array):
        self.memory[pointer][:] = array.view(np.uint8).reshape(-1)

    def copy_out(self, pointer, array):
        array.view(np.uint8).reshape(-1)[:] = self.memory[pointer]

    def tensor(self, **kwargs):
        self.calls.append(("tensor", kwargs))
        return kwargs

    def launch(self, kernel, args, **kwargs):
        self.calls.append(("launch", args, kwargs))
        return kernel(*args)

    def synchronize(self, stream, timeout):
        self.calls.append(("synchronize", stream, timeout))
        if self.fail_sync:
            raise RuntimeError("injected timeout")

    def live_views(self):
        return self.views

    def close(self, pointers, stream):
        self.calls.append(("close", pointers, stream))


@pytest.fixture
def session():
    return NativeSession(rank=0, world_size=2, device=0, bootstrap="tcp://127.0.0.1:25100", control=Control(), bridge=Bridge())


def make_ticket(session, epochs=3):
    data = session.allocate(key="Data", shape=(2, 4), dtype="float32")
    signal = session.allocate(key="Ready", shape=(2,), dtype="int32", kind="signal")
    channel = session.channel(key="comm")
    manifest = {
        "schema_version": 1,
        "profile": NATIVE_PROFILE,
        "groups": [{"name": "world", "size": 2}],
        "channels": [{"name": "comm", "group": "world"}],
        "objects": [
            {
                "key": obj.key,
                "shape": list(obj.shape),
                "dtype": obj.dtype,
                "kind": obj.kind,
                "remote_access": obj.remote_access,
                "strides": list(obj.strides),
                "physical_nbytes": obj.physical_nbytes,
                "alignment": 64,
                "group": "world",
            }
            for obj in (data, signal)
        ],
    }
    kernel = SimpleNamespace(manifest=manifest)
    return session.prepare(kernel, channel=channel, epochs=epochs, protocol="test-v1"), data, signal


def test_bootstrap_metadata_mismatch_never_initializes_device():
    control, bridge = Control(), Bridge()
    original = control.all_gather

    def wrong_device(payload):
        peers = original(payload)
        peers[1]["device"] = 0
        peers[1]["physical_device"] = 0
        return peers

    control.all_gather = wrong_device
    with pytest.raises(ValueError, match="different devices"):
        NativeSession(rank=0, world_size=2, device=0, bootstrap="tcp://127.0.0.1:25100", control=control, bridge=bridge)
    assert bridge.calls == []


def test_allocation_preflight_precedes_any_device_allocation(session):
    session.control.mismatch = True
    with pytest.raises(ValueError, match="metadata mismatch"):
        session.allocate(key="A", shape=(4,), dtype="float32")
    assert not any(call[0] == "allocate" for call in session.bridge.calls)
    assert session.state == "FAILED"


def test_signal_layout_and_logical_readback(session):
    _, _, ready = make_ticket(session)
    assert ready.strides == (16,)
    raw = session.bridge.memory[ready.pointer].view(np.int32)
    raw[:] = -5
    raw[::16] = [1, 2]
    np.testing.assert_array_equal(session.copy_out(ready), [1, 2])
    assert ready.local_tensor["strides"] == (16,)
    # Transport publishes uint32 through the same int32 bit pattern.
    assert ready.view((16,), "uint32", 64)["pointer"] == ready.pointer + 64


def test_copy_in_and_readback_preserve_raw_bfloat16_bits(session):
    data = session.allocate(key="A", shape=(4,), dtype="bfloat16")
    array = np.array([0, 1, 32768, 65535], dtype=np.uint16)
    session.bridge.copy_in(data.pointer, array)
    np.testing.assert_array_equal(session.copy_out(data), array)


def test_view_rejects_out_of_bounds_and_other_rank(session):
    _, data, _ = make_ticket(session)
    with pytest.raises(ValueError, match="exceeds"):
        data.view((9,))
    with pytest.raises(ValueError, match="aligned"):
        data.view((1,), byte_offset=1)
    with pytest.raises(ValueError, match="local rank"):
        data.for_rank(1)
    assert data.view((4,), byte_offset=16)["pointer"] == data.pointer + 16


def test_ticket_copy_and_epoch_reuse(session):
    ticket, data, ready = make_ticket(session)
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    session.copy_in(data, values, ticket=ticket)
    np.testing.assert_array_equal(session.copy_out(data), values)
    with pytest.raises(ValueError, match="signals"):
        session.copy_in(ready, np.zeros(2, dtype=np.int32), ticket=ticket)
    session.mark_running(ticket)
    session.launch_kernel(lambda x: x, 3)
    session.mark_complete(ticket)
    assert session.state == "LOCAL_DONE"
    second = session.prepare(ticket.kernel, channel=ticket.channel, epochs=2, protocol="test-v1")
    assert (second.base, second.end, second.launch_id) == (3, 5, 2)
    with pytest.raises(ValueError, match="stale"):
        session.copy_in(data, values, ticket=ticket)
    with pytest.raises(ValueError, match="stale"):
        session.copy_in(data, values, ticket=replace(second, generation=second.generation + 1))
    session.close()
    session.close()
    assert sum(call[0] == "close" for call in session.bridge.calls) == 1


def test_epoch_overflow_rejected_before_collective(session):
    ticket, _, _ = make_ticket(session)
    session.mark_running(ticket)
    session.mark_complete(ticket)
    session._epoch = MAX_EPOCH
    count = len(session.control.payloads)
    with pytest.raises(OverflowError, match="epochs exhausted"):
        session.prepare(ticket.kernel, channel=ticket.channel, epochs=1, protocol="test-v1")
    assert len(session.control.payloads) == count


def test_failed_sync_keeps_allocations_and_forbids_relaunch(session):
    ticket, _, _ = make_ticket(session)
    session.mark_running(ticket)
    session.bridge.fail_sync = True
    with pytest.raises(RuntimeError, match="timeout"):
        session.mark_complete(ticket)
    assert session.state == "FAILED"
    with pytest.raises(RuntimeError, match="cannot close"):
        session.close()
    with pytest.raises(RuntimeError, match="RUNNING"):
        session.launch_kernel(lambda: None)
    assert not any(call[0] == "close" for call in session.bridge.calls)
    assert len(session.objects) == 2


def test_live_tensor_lease_prevents_heap_free(session):
    session.bridge.views = 1
    with pytest.raises(RuntimeError, match="borrowed"):
        session.close()
    assert not any(call[0] == "close" for call in session.bridge.calls)
    session.bridge.views = 0
    session.close()


def test_copy_input_validation_precedes_transfer(session):
    ticket, data, _ = make_ticket(session)
    for value, kwargs in [
        ({0: np.zeros((2, 4), dtype=np.float32)}, {}),
        (np.zeros((2, 4), dtype=np.float32), {"rank": 1}),
        (np.zeros((2, 4), dtype=np.float32), {"stream": 123}),
        (np.zeros((2, 4), dtype=np.float64), {}),
    ]:
        with pytest.raises(ValueError):
            session.copy_in(data, value, ticket=ticket, **kwargs)
    assert not session.bridge.memory[data.pointer].any()


@pytest.mark.parametrize("bootstrap", ["tcp://0.0.0.0:22", "tcp://127.0.0.1", "http://127.0.0.1:22", "tcp://127.0.0.1:22/path"])
def test_native_bootstrap_is_single_host_only(bootstrap):
    with pytest.raises(ValueError, match="bootstrap"):
        _loopback_bootstrap(bootstrap)


def test_factory_does_not_need_torch_npu_with_injected_transport():
    ctx = init_runtime(rank=0, device=0, bootstrap="tcp://127.0.0.1:25100", control=Control(), bridge=Bridge())
    assert ctx.profile == NATIVE_PROFILE
    ctx.close()


def test_manifest_mismatch_rejected_before_launch(session):
    ticket, _, _ = make_ticket(session)
    session.state = "QUIESCENT"
    bad = copy.deepcopy(ticket.kernel.manifest)
    bad["objects"][0]["shape"] = [4, 2]
    with pytest.raises(ValueError, match="mismatched shape"):
        session.prepare(SimpleNamespace(manifest=bad), channel=ticket.channel, epochs=1, protocol="test")
    assert not any(call[0] == "launch" for call in session.bridge.calls)


def test_concurrent_prepare_reserves_only_one_ticket(session):
    from concurrent.futures import ThreadPoolExecutor

    first, _, _ = make_ticket(session)
    session.mark_running(first)
    session.mark_complete(first)

    def prepare():
        try:
            return session.prepare(first.kernel, channel=first.channel, epochs=1, protocol="test-v1")
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: prepare(), range(2)))
    assert sum(ticket is not None for ticket in results) == 1
    assert session._epoch == 4


def test_host_bridge_cache_invalidates_sdk_change(tmp_path, monkeypatch):
    import tilelang.ascend.distributed.native_runtime as runtime

    root, cann = tmp_path / "shmem", tmp_path / "cann"
    root.mkdir()
    headers = cann / "include/acl"
    headers.mkdir(parents=True)
    header = headers / "acl_rt.h"
    header.write_text("first SDK revision")
    library = root / "libshmem.so"
    library.write_bytes(b"library")
    builds = []

    def check_output(command, **kwargs):
        return runtime.SHMEM_REVISION if command[0] == "git" else "test host compiler"

    def run(command, **kwargs):
        if command[0] != "git":
            from pathlib import Path

            builds.append(command)
            Path(command[-1]).write_bytes(b"compiled")

    monkeypatch.setattr(runtime.subprocess, "check_output", check_output)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/c++")
    args = dict(shmem_root=root, shmem_library=library, cann_root=cann, output_dir=tmp_path / "cache")
    first, _ = runtime.build_host_bridge(**args)
    again, _ = runtime.build_host_bridge(**args)
    assert first == again and len(builds) == 1
    header.write_text("different SDK revision")
    changed, _ = runtime.build_host_bridge(**args)
    assert changed != first and len(builds) == 2


def test_real_bridge_session_is_process_singleton(monkeypatch):
    import tilelang.ascend.distributed.native_runtime as runtime

    class RuntimeBridge(Bridge, runtime.ShmemBridge):
        def device_identity(self, device):
            return {"physical_device": device, "soc": "fake"}

    monkeypatch.setattr(runtime, "_ACTIVE_NATIVE_SESSION", None)

    def init():
        return runtime.init_runtime(rank=0, device=0, bootstrap="tcp://127.0.0.1:25100", control=Control(), bridge=RuntimeBridge())

    first = init()
    with pytest.raises(RuntimeError, match="one native SHMEM session"):
        init()
    first.close()
    second = init()
    second.fail("device timeout")
    with pytest.raises(RuntimeError, match="one native SHMEM session"):
        init()


def test_real_bridge_dlpack_metadata_and_lease():
    """Opt-in SDK-only ABI probe; never initialize ACL or dereference an NPU pointer."""
    import gc
    import os

    if os.environ.get("TILELANG_TEST_SHMEM_BRIDGE") != "1":
        pytest.skip("set TILELANG_TEST_SHMEM_BRIDGE=1 with the pinned SHMEM build to run the C++ ABI probe")
    from tilelang.ascend.distributed.native_runtime import ShmemBridge

    bridge = ShmemBridge(shmem_root=os.environ["SHMEM_ROOT"])
    backing = np.zeros(32, dtype=np.int32)
    assert bridge.live_views() == 0
    tensor = bridge.tensor(pointer=backing.ctypes.data, device=1, shape=(2,), dtype="int32", strides=(16,))
    assert tensor.shape == (2,) and tensor.strides == (16,)
    assert str(tensor.dtype) == "int32" and tensor.device.dlpack_device_type() == 12 and tensor.device.index == 1
    assert bridge.live_views() == 1
    del tensor
    gc.collect()
    assert bridge.live_views() == 0


def test_real_tcpstore_metadata_collective():
    from concurrent.futures import ThreadPoolExecutor
    import socket
    from threading import Barrier

    from tilelang.ascend.distributed.native_runtime import StoreControl

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    finished = Barrier(2)

    def worker(rank):
        control = StoreControl(rank=rank, world_size=2, port=port, timeout=10)
        first = control.all_gather({"rank": rank, "manifest": "a"})
        second = control.all_gather({"epoch": 7})
        finished.wait(timeout=10)  # Keep the server alive until both clients finish reading.
        return first, second

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(worker, range(2)))
    expected = ([{"rank": 0, "manifest": "a"}, {"rank": 1, "manifest": "a"}], [{"epoch": 7}, {"epoch": 7}])
    assert reports == [expected, expected]


@pytest.mark.parametrize("offset", [1, 2, None])
def test_prepare_defers_additional_scalar_binding_until_launch(session, tmp_path, monkeypatch, offset):
    import tilelang.ascend.language as T
    import tilelang.distributed as D
    import tilelang.language.distributed as TD
    from tilelang.ascend.distributed.native import NativeCompiledProgram
    from testing.ascend.distributed.test_native_plan import fake_headers

    group = TD.GroupSpec("world", 2)
    channel_spec = TD.ChannelSpec("comm", group)

    @T.prim_func
    def program(Ready: TD.SignalBuffer((1,), group=group), epoch0: T.int32, offset: T.int32):
        with T.Kernel(1):
            peer = 1 - TD.rank(group)
            TD.signal(TD.remote(Ready, peer)[0], epoch0 + offset, channel=channel_spec)
            TD.wait_signal(Ready[0], epoch0 + offset, channel=channel_spec, covers=[])

    kernel = NativeCompiledProgram(program, shmem_root=fake_headers(tmp_path), output_dir=tmp_path / "runs")
    session.allocate(key="Ready", shape=(1,), dtype="int32", kind="signal")
    channel = session.channel(key="comm")
    ticket = session.prepare(kernel, channel=channel, epochs=1, protocol="scalar-v1")
    assert session.state == "PREPARED" and ticket.base == 0 and ticket.end == 1
    assert not any(call[0] == "launch" for call in session.bridge.calls)

    def build(**kwargs):
        for rank, plan in kwargs["plans"].items():
            for instruction in plan:
                spec = kernel._transport_spec(instruction, rank)
                if spec is not None:
                    kind, nbytes = spec
                    kernel._kernels[("transport", kind, nbytes, kernel.header_fingerprint, kernel.transport_compile_profile)] = (
                        lambda *args: None
                    )

    monkeypatch.setattr(kernel, "build", build)
    if offset == 1:
        result = D.complete(D.launch(kernel, ticket=ticket, scalars={"offset": offset}))
        assert result["passed"] and session.state == "LOCAL_DONE"
        assert sum(call[0] == "launch" for call in session.bridge.calls) == 2
        session.close()
    else:
        message = "missing a scalar" if offset is None else "outside the prepared native interval"
        with pytest.raises(ValueError, match=message):
            D.launch(kernel, ticket=ticket, scalars={} if offset is None else {"offset": offset})
        assert session.state == "FAILED"
        assert not any(call[0] == "launch" for call in session.bridge.calls)


def test_prepare_without_extra_scalars_preserves_early_epoch_validation(session):
    import tilelang.ascend.language as T
    import tilelang.language.distributed as TD
    from tilelang.ascend.distributed.native import NativeCompiledProgram

    group = TD.GroupSpec("world", 2)
    channel_spec = TD.ChannelSpec("comm", group)

    @T.prim_func
    def program(Ready: TD.SignalBuffer((1,), group=group), epoch0: T.int32):
        with T.Kernel(1):
            TD.signal(TD.remote(Ready, 1 - TD.rank(group))[0], epoch0 + 2, channel=channel_spec)

    kernel = NativeCompiledProgram(program)
    session.allocate(key="Ready", shape=(1,), dtype="int32", kind="signal")
    channel = session.channel(key="comm")
    with pytest.raises(ValueError, match="outside the prepared native interval"):
        session.prepare(kernel, channel=channel, epochs=1, protocol="too-short")
    assert session.state == "QUIESCENT" and session._ticket is None and session._epoch == 0
    with pytest.raises(RuntimeError, match="RUNNING"):
        session.launch_kernel(lambda: None)
    assert not any(call[0] == "launch" for call in session.bridge.calls)
