"""Exercise distributed JIT binding and real phase execution without an NPU.

Only compilation and the SHMEM device boundary are replaced. The tests retain
the source frontend, native planner, Session state machine and native runner.
"""

import copy
from dataclasses import dataclass
from threading import Condition, Thread

import numpy as np
import pytest

import tilelang.ascend.language as T
import tilelang.distributed as D
import tilelang.language.distributed as TD
from tilelang.ascend.distributed.native import NativeCompiledProgram
from tilelang.ascend.distributed.native_runtime import NativeSession
from testing.ascend.distributed.test_native_plan import fake_headers, transfer
from testing.ascend.distributed.test_native_runtime import Bridge, Control


GROUP = TD.GroupSpec("world", 2)
CHANNEL = TD.ChannelSpec("comm", GROUP)


def copy_program(size=8, dtype="float32"):
    @T.prim_func
    def program(
        Src: TD.SymBuffer((size,), dtype, group=GROUP, remote_access="read"),
        Dst: TD.SymBuffer((size,), dtype, group=GROUP, remote_access="write"),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            T.copy(Src, Dst)
            TD.quiet(channel=CHANNEL)

    return program


def inout_program(size=8):
    @T.prim_func
    def program(
        Data: TD.SymBuffer((size,), "float32", group=GROUP, remote_access="read_write"),
        Scratch: TD.SymBuffer((size,), "float32", group=GROUP, remote_access="read_write"),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            T.copy(Data, Scratch)
            T.copy(Scratch, Data)
            TD.quiet(channel=CHANNEL)

    return program


def history_program():
    @T.prim_func
    def program(Ready: TD.SignalBuffer((1,), group=GROUP), epoch0: T.int32, use_history: T.int32):
        with T.Kernel(1):
            if use_history:
                TD.wait_signal(Ready[0], epoch0, channel=CHANNEL, covers=[])
            TD.signal(TD.remote(Ready, 1 - TD.rank(GROUP))[0], epoch0 + 1, channel=CHANNEL)
            TD.quiet(channel=CHANNEL)

    return program


@dataclass(frozen=True)
class CPUTransport:
    kind: str
    nbytes: int


class Fabric:
    def __init__(self):
        self.condition = Condition()
        self.bridges = {}
        self.collectives = {}


class PairedControl:
    """A bounded, real two-participant collective, including unequal payloads."""

    def __init__(self, fabric, rank):
        self.fabric, self.rank, self.sequence = fabric, rank, 0

    def all_gather(self, payload):
        with self.fabric.condition:
            entries = self.fabric.collectives.setdefault(self.sequence, {})
            self.sequence += 1
            entries[self.rank] = copy.deepcopy(payload)
            self.fabric.condition.notify_all()
            if not self.fabric.condition.wait_for(lambda: len(entries) == 2, timeout=3):
                raise TimeoutError("test collective did not receive both ranks")
            return [copy.deepcopy(entries[rank]) for rank in range(2)]


class CPUBridge(Bridge):
    def __init__(self, rank=0, fabric=None):
        super().__init__()
        self.rank, self.fabric = rank, fabric or Fabric()
        self.fabric.bridges[rank] = self
        self.fail_copy = False
        self.fail_readback = False

    def copy_in(self, pointer, array):
        self.calls.append(("copy_in", pointer))
        if self.fail_copy:
            raise RuntimeError("injected input transfer failure")
        super().copy_in(pointer, array)

    def copy_out(self, pointer, array):
        if self.fail_readback:
            raise RuntimeError("injected output transfer failure")
        super().copy_out(pointer, array)

    def _bytes(self, view, count):
        pointer = view["pointer"]
        for base, allocation in self.memory.items():
            offset = pointer - base
            if offset >= 0 and offset + count <= allocation.size:
                return allocation[offset : offset + count]
        raise AssertionError(f"unmapped fake device pointer {pointer}")

    def launch(self, kernel, args, **kwargs):
        self.calls.append(("launch", kernel.kind, args))
        with self.fabric.condition:
            if kernel.kind in ("local_copy", "put", "get", "put_signal"):
                dst, src, peer, *signal = args
                source = self.fabric.bridges[peer] if kernel.kind == "get" else self
                destination = self if kernel.kind in ("get", "local_copy") else self.fabric.bridges[peer]
                destination._bytes(dst, kernel.nbytes)[:] = source._bytes(src, kernel.nbytes)
                if signal:
                    destination._bytes(signal[0], 4).view(np.uint32)[0] = signal[1]
            elif kernel.kind == "signal":
                signal, peer, value = args
                self.fabric.bridges[peer]._bytes(signal, 4).view(np.uint32)[0] = value
            elif kernel.kind == "wait_signal":
                signal, value = args
                if not self.fabric.condition.wait_for(lambda: self._bytes(signal, 4).view(np.uint32)[0] >= value, timeout=3):
                    raise TimeoutError("test device signal was never published")
            else:
                raise AssertionError(f"unexpected transport {kernel.kind}")
            self.fabric.condition.notify_all()


def parallel(*functions):
    results, errors = [None] * len(functions), [None] * len(functions)

    def invoke(index, function):
        try:
            results[index] = function()
        except BaseException as error:
            errors[index] = error

    threads = [Thread(target=invoke, args=(i, function), daemon=True) for i, function in enumerate(functions)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads), "distributed caller did not terminate"
    return results, errors


def successful(*functions):
    results, errors = parallel(*functions)
    assert errors == [None] * len(functions), errors
    return results


def assert_collective_errors(errors):
    assert all(isinstance(error, Exception) for error in errors), errors
    assert not any(isinstance(error, TimeoutError) for error in errors), errors


@pytest.fixture
def jit_factory(tmp_path, monkeypatch):
    headers = fake_headers(tmp_path)

    def build(compiled, *, plans=None, scalars=None, epochs=None, **kwargs):
        plans = plans if plans is not None else compiled.plan(scalars=scalars, epoch_end=epochs)
        transports, segments = compiled._unique(plans)
        assert not segments, "these lifecycle regressions require no GEMM emulation"
        for kind, nbytes in transports:
            key = ("transport", kind, nbytes, compiled.header_fingerprint, compiled.transport_compile_profile)
            compiled._kernels.setdefault(key, CPUTransport(kind, nbytes))
        return compiled._kernels

    monkeypatch.setattr(NativeCompiledProgram, "build", build)

    def decorate(factory=copy_program, *, inputs=("Src",), outputs=("Dst",), epochs=1):
        return D.jit(inputs=inputs, outputs=outputs, epochs=epochs, shmem_root=headers, output_dir=tmp_path / "runs")(factory)

    decorate.headers = headers
    return decorate


@pytest.fixture
def session(jit_factory):
    return NativeSession(
        rank=0,
        world_size=2,
        device=0,
        bootstrap="tcp://127.0.0.1:25100",
        control=Control(),
        bridge=CPUBridge(),
        shmem_root=jit_factory.headers,
    )


def paired_sessions(headers, *, heap_bytes=1024 * 1024):
    fabric = Fabric()
    bridges = [CPUBridge(rank, fabric) for rank in range(2)]

    def initialize(rank):
        return NativeSession(
            rank=rank,
            world_size=2,
            device=rank,
            bootstrap="tcp://127.0.0.1:25100",
            control=PairedControl(fabric, rank),
            bridge=bridges[rank],
            shmem_root=headers,
            heap_bytes=heap_bytes,
        )

    return successful(lambda: initialize(0), lambda: initialize(1))


def test_factory_defaults_and_named_arguments_share_specialization(jit_factory):
    factory = jit_factory()
    assert factory() is factory(8) is factory(size=8) is factory(dtype="float32", size=8)
    assert factory(16) is not factory()
    with pytest.raises(AttributeError):
        factory().prim_func = copy_program(16)


def test_static_cache_preserves_python_argument_types(jit_factory):
    def typed_factory(marker=1):
        return copy_program(8 if type(marker) is int else 16)

    factory = jit_factory(typed_factory)
    assert factory(1) is not factory(True)
    assert factory(1) is not factory(1.0)
    assert factory(True) is not factory(1.0)


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [(("Src", "Src"), ("Dst",)), (("Src",), ("Dst", "Dst")), (("missing",), ("Dst",))],
)
def test_roles_require_declared_unique_buffer_names(jit_factory, inputs, outputs):
    with pytest.raises(ValueError):
        jit_factory(inputs=inputs, outputs=outputs)()


def test_repeated_calls_reuse_storage_and_return_independent_outputs(jit_factory, session):
    kernel = jit_factory()()
    bound = kernel.bind(session)
    first_input = np.arange(8, dtype=np.float32)
    first = bound(first_input)
    allocations = sum(call[0] == "allocate" for call in session.bridge.calls)
    second = bound(Src=first_input + 10)
    np.testing.assert_array_equal(first, first_input)
    np.testing.assert_array_equal(second, first_input + 10)
    assert not np.shares_memory(first, second)
    assert bound.last_run.completed
    assert bound.last_run.ticket.launch_id == 2
    assert (bound.last_run.ticket.base, bound.last_run.ticket.end) == (1, 2)
    assert kernel.bind(session).binding is bound.binding
    assert sum(call[0] == "allocate" for call in session.bridge.calls) == allocations
    session.close()
    np.testing.assert_array_equal(first, first_input)


def test_multiple_shapes_share_session_without_aliasing_workspaces(jit_factory, session):
    factory = jit_factory()
    small = factory().bind(session)
    np.testing.assert_array_equal(small(np.ones(8, dtype=np.float32)), 1)
    large = factory(16).bind(session)
    np.testing.assert_array_equal(large(np.full(16, 2, dtype=np.float32)), 2)
    pointers = [{obj.pointer for obj in binding.objects.values()} for binding in (small.binding, large.binding)]
    assert pointers[0].isdisjoint(pointers[1])
    allocations = sum(call[0] == "allocate" for call in session.bridge.calls)
    np.testing.assert_array_equal(small(np.full(8, 3, dtype=np.float32)), 3)
    assert sum(call[0] == "allocate" for call in session.bridge.calls) == allocations
    assert session._launch_id == 3
    session.close()


def test_manual_launch_defaults_keep_working_after_jit_workspace_allocation(jit_factory, session, tmp_path):
    compiled = NativeCompiledProgram(copy_program(), shmem_root=jit_factory.headers, output_dir=tmp_path / "manual", debug=False)
    objects = {
        spec["key"]: session.allocate(
            key=spec["key"], shape=spec["shape"], dtype=spec["dtype"], remote_access=spec["remote_access"], kind=spec["kind"]
        )
        for spec in compiled.manifest["objects"]
    }
    channel = session.channel(key="comm")

    def manual_call(values):
        ticket = session.prepare(compiled, channel=channel, epochs=1, protocol="manual-copy")
        session.copy_in(objects["Src"], values, ticket=ticket)
        run = D.launch(compiled, ticket=ticket)
        D.complete(run)
        return session.copy_out(objects["Dst"])

    values = np.arange(8, dtype=np.float32)
    np.testing.assert_array_equal(manual_call(values), values)
    bound = jit_factory()(16).bind(session)
    np.testing.assert_array_equal(bound(np.full(16, 7, dtype=np.float32)), 7)
    np.testing.assert_array_equal(manual_call(values + 20), values + 20)
    assert session._launch_id == 3
    session.close()


def test_inout_role_and_multiple_outputs_preserve_input_values(jit_factory, session):
    bound = jit_factory(inout_program, inputs=("Data",), outputs=("Data", "Scratch"))().bind(session)
    original = np.arange(8, dtype=np.float32)
    first, scratch = bound(original)
    np.testing.assert_array_equal(first, original)
    np.testing.assert_array_equal(scratch, original)
    first[:] = -1
    np.testing.assert_array_equal(scratch, original)
    np.testing.assert_array_equal(original, np.arange(8, dtype=np.float32))
    later, _ = bound(original + 5)
    np.testing.assert_array_equal(later, original + 5)
    np.testing.assert_array_equal(first, -1)


def test_bfloat16_inputs_keep_raw_uint16_storage(jit_factory, session):
    bound = jit_factory()(size=16, dtype="bfloat16").bind(session)
    bits = np.arange(16, dtype=np.uint16) * 1024
    np.testing.assert_array_equal(bound(bits), bits)


def test_strided_numpy_input_is_packed_without_mutating_the_original(jit_factory, session):
    bound = jit_factory()().bind(session)
    storage = np.arange(16, dtype=np.float32)
    np.testing.assert_array_equal(bound(storage[::2]), np.arange(0, 16, 2, dtype=np.float32))
    np.testing.assert_array_equal(storage, np.arange(16, dtype=np.float32))


@pytest.mark.parametrize("value", [np.ones(8, dtype=np.float64), np.ones(16, dtype=np.float32), [1.0] * 8])
def test_invalid_input_never_transfers_or_launches(jit_factory, session, value):
    bound = jit_factory()().bind(session)
    before = len(session.bridge.calls)
    with pytest.raises((TypeError, ValueError)):
        bound(value)
    assert not any(call[0] in ("copy_in", "launch") for call in session.bridge.calls[before:])


@pytest.mark.parametrize("scalars", [{"epoch0": 0}, {"missing": 1}, {"use_history": 1.5}])
def test_invalid_launch_scalars_never_launch(jit_factory, session, scalars):
    bound = jit_factory()().bind(session)
    with pytest.raises((TypeError, ValueError)):
        bound(np.ones(8, dtype=np.float32), scalars=scalars)
    assert not any(call[0] == "launch" for call in session.bridge.calls)


def test_static_epoch_parameter_uses_its_factory_default(jit_factory, session):
    def program(rounds=3):
        return copy_program()

    bound = jit_factory(program, epochs="rounds")().bind(session)
    bound(np.ones(8, dtype=np.float32))
    assert bound.last_run.ticket.epochs == 3


def test_closed_session_cannot_reuse_bound_callable(jit_factory, session):
    bound = jit_factory()().bind(session)
    session.close()
    before = len(session.bridge.calls)
    with pytest.raises((RuntimeError, ValueError)):
        bound(np.ones(8, dtype=np.float32))
    assert not any(call[0] in ("copy_in", "launch") for call in session.bridge.calls[before:])


def test_two_rank_transport_returns_peer_data_and_reuses_signals(jit_factory):
    sessions = paired_sessions(jit_factory.headers)
    kernel = jit_factory(transfer, inputs=("Send",), outputs=("Recv",))()
    bound = successful(lambda: kernel.bind(sessions[0]), lambda: kernel.bind(sessions[1]))
    for iteration in range(2):
        arrays = [np.arange(16, dtype=np.float32) + 100 * rank + iteration for rank in range(2)]
        outputs = successful(lambda arrays=arrays: bound[0](arrays[0]), lambda arrays=arrays: bound[1](arrays[1]))
        for rank in range(2):
            np.testing.assert_array_equal(outputs[rank][:8], arrays[1 - rank][:8])
            assert bound[rank].binding.signal_values
    successful(sessions[0].close, sessions[1].close)


def test_one_rank_input_error_is_reported_without_peer_timeout(jit_factory):
    sessions = paired_sessions(jit_factory.headers)
    kernel = jit_factory()()
    bound = successful(lambda: kernel.bind(sessions[0]), lambda: kernel.bind(sessions[1]))
    _, errors = parallel(lambda: bound[0](np.ones(8, dtype=np.float64)), lambda: bound[1](np.ones(8, dtype=np.float32)))
    assert_collective_errors(errors)
    assert all("rank 0" in str(error) and "expects" in str(error) and "float32" in str(error) for error in errors)
    assert not any(call[0] == "launch" for session in sessions for call in session.bridge.calls)


def test_one_rank_transfer_failure_keeps_allocations_and_stops_peer_launch(jit_factory):
    sessions = paired_sessions(jit_factory.headers)
    kernel = jit_factory()()
    bound = successful(lambda: kernel.bind(sessions[0]), lambda: kernel.bind(sessions[1]))
    sessions[0].bridge.fail_copy = True
    _, errors = parallel(lambda: bound[0](np.ones(8, dtype=np.float32)), lambda: bound[1](np.ones(8, dtype=np.float32)))
    assert_collective_errors(errors)
    assert all(session.state == "FAILED" and len(session.objects) == 2 for session in sessions)
    assert not any(call[0] in ("launch", "close") for session in sessions for call in session.bridge.calls)


def test_one_rank_output_failure_does_not_return_success_on_its_peer(jit_factory):
    sessions = paired_sessions(jit_factory.headers)
    kernel = jit_factory()()
    bound = successful(lambda: kernel.bind(sessions[0]), lambda: kernel.bind(sessions[1]))
    sessions[0].bridge.fail_readback = True
    _, errors = parallel(lambda: bound[0](np.ones(8, dtype=np.float32)), lambda: bound[1](np.ones(8, dtype=np.float32)))
    assert_collective_errors(errors)
    assert all(session.state == "FAILED" and len(session.objects) == 2 for session in sessions)
    assert not any(call[0] == "close" for session in sessions for call in session.bridge.calls)


def test_rank_specialization_mismatch_fails_before_allocation(jit_factory):
    sessions = paired_sessions(jit_factory.headers)
    factory = jit_factory()
    _, errors = parallel(lambda: factory(8).bind(sessions[0]), lambda: factory(16).bind(sessions[1]))
    assert_collective_errors(errors)
    assert not any(call[0] == "allocate" for session in sessions for call in session.bridge.calls)


def test_binding_cache_hit_mismatch_never_starts_new_collective_allocation(jit_factory):
    sessions = paired_sessions(jit_factory.headers)
    kernel = jit_factory()()
    successful(lambda: kernel.bind(sessions[0]), lambda: kernel.bind(sessions[1]))
    sessions[1]._bindings.clear()
    before = [len(session.bridge.calls) for session in sessions]
    _, errors = parallel(lambda: kernel.bind(sessions[0]), lambda: kernel.bind(sessions[1]))
    assert_collective_errors(errors)
    assert not any(call[0] == "allocate" for session, start in zip(sessions, before) for call in session.bridge.calls[start:])


def test_new_binding_cannot_wait_for_an_epoch_published_by_another_binding(jit_factory):
    sessions = paired_sessions(jit_factory.headers)
    kernel = jit_factory(history_program, inputs=(), outputs=())()
    first = successful(lambda: kernel.bind(sessions[0]), lambda: kernel.bind(sessions[1]))
    assert successful(lambda: first[0](scalars={"use_history": 0}), lambda: first[1](scalars={"use_history": 0})) == [None, None]
    assert successful(lambda: first[0](scalars={"use_history": np.int32(1)}), lambda: first[1](scalars={"use_history": 1})) == [None, None]
    fresh = successful(lambda: kernel.bind(sessions[0], key="fresh"), lambda: kernel.bind(sessions[1], key="fresh"))
    before = [len(session.bridge.calls) for session in sessions]
    _, errors = parallel(lambda: fresh[0](scalars={"use_history": 1}), lambda: fresh[1](scalars={"use_history": 1}))
    assert_collective_errors(errors)
    assert any("deadlock" in str(error) or "signal" in str(error) for error in errors)
    assert not any(call[0] == "launch" for session, start in zip(sessions, before) for call in session.bridge.calls[start:])


def test_heap_exhaustion_is_collective_and_preserves_existing_binding(jit_factory):
    sessions = paired_sessions(jit_factory.headers, heap_bytes=64)
    factory = jit_factory()
    bound = successful(lambda: factory(8).bind(sessions[0]), lambda: factory(8).bind(sessions[1]))
    before = [len(session.bridge.calls) for session in sessions]
    _, errors = parallel(lambda: factory(16).bind(sessions[0]), lambda: factory(16).bind(sessions[1]))
    assert_collective_errors(errors)
    assert all(len(session.objects) == 2 for session in sessions)
    assert not any(call[0] in ("allocate", "close") for session, start in zip(sessions, before) for call in session.bridge.calls[start:])
    outputs = successful(lambda: bound[0](np.ones(8, dtype=np.float32)), lambda: bound[1](np.full(8, 2, dtype=np.float32)))
    np.testing.assert_array_equal(outputs[0], 1)
    np.testing.assert_array_equal(outputs[1], 2)
    successful(sessions[0].close, sessions[1].close)
