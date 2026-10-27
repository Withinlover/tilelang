"""Single-kernel planning, ABI and compiler checks; these do not execute NPUs."""

import pytest

import tilelang
import tilelang.distributed as D
from tilelang.ascend.distributed.fused import FusedCompiledProgram, verify_fused_protocol
from examples.distributed.example_allgather_gemm import allgather_gemm
from examples.distributed.example_fused_exchange import vector_exchange
from examples.distributed.run_native_distributed import exchange
from testing.ascend.distributed.test_native_plan import fake_headers
from testing.ascend.distributed.test_native_runtime import Bridge, Control


@pytest.mark.parametrize("rounds,slots", [(1, 2), (3, 2), (7, 1)])
@pytest.mark.parametrize("case", ["put", "get", "vector", "gemm"])
def test_fused_supported_plans_and_epoch_translation(rounds, slots, case):
    program = (
        allgather_gemm(rounds, slots)
        if case == "gemm"
        else vector_exchange(rounds, slots)
        if case == "vector"
        else exchange(rounds, slots, get=case == "get")
    )
    profile = D.P0B_PROFILE if case == "gemm" else D.P0A_PROFILE
    kernel = D.compile(program, profile=profile)
    first = kernel.plan(epoch_end=rounds)
    second = kernel.plan(epoch_base=rounds, epoch_end=2 * rounds)
    assert len(first[0]) == len(second[0])
    assert kernel.manifest["descriptor"]["encoding"] == "single-kernel-shmem-v1"
    assert kernel.manifest["execution"] == ("native-single-kernel-p0b" if case == "gemm" else "native-single-kernel-p0a")
    for rank in (0, 1):
        for before, after in zip(first[rank], second[rank]):
            if before[0] in ("signal", "wait_signal"):
                assert after[2] == before[2] + rounds


@pytest.mark.parametrize(
    "fault,diagnostic",
    [
        ("missing_acquire", "READY/acquire"),
        ("missing_release", "release"),
        ("early_ack", "READY/acquire|unordered access/slot reuse"),
    ],
)
def test_fused_protocol_rejects_unsafe_receive_lifetime(fault, diagnostic):
    kernel = FusedCompiledProgram(allgather_gemm(3, 1), profile=D.P0B_PROFILE)
    plans = kernel.plan()
    changed = []
    for instruction in plans[0]:
        if fault == "missing_acquire" and instruction[0] == "wait_signal" and instruction[3]:
            instruction = (*instruction[:3], ())
        if fault == "missing_release" and instruction[0] == "signal":
            instruction = (*instruction[:3], ())
        changed.append(instruction)
    plans[0] = changed
    if fault == "early_ack":
        ack = next(i for i, op in enumerate(changed) if op[0] == "signal")
        receive_compute = ack - 1
        changed[receive_compute], changed[ack] = changed[ack], changed[receive_compute]
    with pytest.raises(ValueError, match=diagnostic):
        verify_fused_protocol(plans)


def test_p0a_rejects_cube_and_p0b_rejects_vector():
    with pytest.raises(ValueError, match="pure AIV"):
        D.compile(allgather_gemm(1, 1), profile=D.P0A_PROFILE).plan()
    with pytest.raises(ValueError, match="vector arithmetic uses P0-A"):
        D.compile(vector_exchange(1, 1), profile=D.P0B_PROFILE).plan()


@pytest.mark.parametrize("case", ["put", "vector", "gemm"])
def test_fused_lower_single_entry_and_protocol_source(case, tmp_path):
    program = allgather_gemm(1, 1) if case == "gemm" else vector_exchange(1, 1) if case == "vector" else exchange(1, 1)
    profile = D.P0B_PROFILE if case == "gemm" else D.P0A_PROFILE
    kernel = D.compile(program, profile=profile)
    result = kernel.lower(output_dir=tmp_path)
    assert result["device_executed"] is False
    for artifact in result["artifacts"]:
        source = (tmp_path / artifact["source"]).read_text()
        assert source.count("__global__") == 1
        assert "distributed.h" in source
        assert "td_fused_" in source
        if case == "gemm":
            assert "__mix__" in source
            assert "asc_mmad" in source
        else:
            assert "__vector__" in source
        if case == "vector":
            assert "vadd" in source.lower() or "asc_add" in source


@pytest.mark.parametrize("profile", [D.P0A_PROFILE, D.P0B_PROFILE])
def test_fused_session_one_launch_physical_signal_abi_and_cache(monkeypatch, tmp_path, profile):
    root = fake_headers(tmp_path)
    kernel = D.compile(exchange(1, 2), profile=profile, shmem_root=root, debug=False)
    built, submitted = [], []

    def compile_fake(func, **kwargs):
        built.append(func)
        return lambda *args: submitted.append(args)

    monkeypatch.setattr(tilelang, "compile", compile_fake)
    session = D.init(
        profile=profile, rank=0, device=0, bootstrap="tcp://127.0.0.1:25100", control=Control(), bridge=Bridge(), shmem_root=root
    )
    for spec in kernel.manifest["objects"]:
        common = dict(key=spec["key"], shape=spec["shape"], dtype=spec["dtype"])
        (
            D.alloc_signals(session, **common)
            if spec["kind"] == "signal"
            else D.alloc_symmetric(session, remote_access=spec["remote_access"], **common)
        )
    channel = D.channel(session, key="comm")
    for base in (0, 1):
        ticket = D.prepare_launch(session, kernel, channel=channel, epochs=1, protocol="fused-abi")
        result = D.complete(D.launch(kernel, ticket=ticket))
        assert result["physical_kernel_count"] == 1
        assert result["timing_basis"] == "host_wall"
        timings = result["timing_seconds"]
        assert set(timings) == {
            "validate_bindings",
            "plan_validation",
            "plan_preflight",
            "build_or_cache_validation",
            "build_preflight",
            "bind_and_submit",
            "stream_sync",
            "report_finalize",
            "total",
        }
        assert all(value >= 0 for value in timings.values())
        assert timings["total"] >= sum(value for key, value in timings.items() if key != "total")
        assert submitted[-1][-1] == base
        views = submitted[-1][:-1]
        assert len(views) == len(kernel.manifest["objects"])
        for spec, view in zip(kernel.manifest["objects"], views):
            if spec["kind"] == "signal":
                assert view["dtype"] == "uint32"
                assert view["shape"] == (32,)
    assert len(submitted) == 2
    assert len(built) == 2  # One per rank, reused across epochs.
    D.close(session)


def test_fused_get_source_lease_rejects_ack_before_get():
    kernel = D.compile(exchange(3, 1, get=True), profile=D.P0A_PROFILE)
    plans = kernel.plan()
    rank0 = plans[0]
    fetched = next(i for i, item in enumerate(rank0) if item[0] == "copy_async")
    ack = next(i for i, item in enumerate(rank0) if item[0] == "signal" and item[1].key == "Ack")
    rank0.insert(fetched, rank0.pop(ack))
    with pytest.raises(ValueError, match="unordered access/slot reuse"):
        verify_fused_protocol(plans)


def test_fused_helper_change_invalidates_bound_manifest(monkeypatch, tmp_path):
    from importlib import import_module

    env = import_module("tilelang.env")

    template = tmp_path / "templates"
    helper = template / "tl_templates/ascend/distributed.h"
    helper.parent.mkdir(parents=True)
    helper.write_text("first helper identity")
    monkeypatch.setattr(env, "TILELANG_TEMPLATE_PATH", str(template))
    kernel = D.compile(exchange(1, 1), profile=D.P0A_PROFILE)
    kernel.plan()
    helper.write_text("changed helper identity")
    with pytest.raises(ValueError, match="fusion device helper changed"):
        kernel.plan()


def test_fused_vector_rejects_untracked_gm_read():
    from tvm import tirx
    from tilelang.ascend.distributed.fused import _verify_vector_memory

    gm = tirx.decl_buffer((64,), "float32", name="received", scope="global")
    block = tirx.SBlock([], [], [], "SIMD_VF", tirx.Evaluate(gm[0] + 1))
    with pytest.raises(ValueError, match="direct/GM loads"):
        _verify_vector_memory(block, [])


def test_p0a_rejects_mask_lane_count_used_as_element_width():
    from tvm import tirx

    program = vector_exchange(1, 1)

    def wrong_width(node):
        if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == "tl.simd.pset":
            return tirx.Call(node.dtype, node.op, [tirx.const(64, "int32"), node.args[1]], annotations=node.annotations)

    body = tirx.stmt_functor.ir_transform(program.body, wrong_width, None)
    with pytest.raises(ValueError, match="element width, not lane count"):
        D.compile(program.with_body(body), profile=D.P0A_PROFILE).plan()


def test_fused_rejects_zero_work():
    import tilelang.ascend.language as T
    import tilelang.language.distributed as TD

    group = TD.GroupSpec("world", 2)
    channel = TD.ChannelSpec("comm", group)

    @T.prim_func
    def empty(Data: TD.SymBuffer((8,), "float32", group=group, remote_access="read_write"), epoch0: T.int32):
        with T.Kernel(1):
            TD.quiet(channel=channel)

    with pytest.raises(ValueError, match="skip zero work on the host"):
        D.compile(empty, profile=D.P0A_PROFILE).plan()


def test_fused_vector_rejects_out_of_bounds_lanes():
    from tvm import tirx
    from tilelang import tvm
    from tilelang.ascend.distributed.fused import _verify_vector_memory

    ub = tirx.decl_buffer((64, 64), "float32", name="local", scope="shared.dyn")
    pointer = tirx.call_intrin("handle", tvm.ir.Op.get("tl.access_ptr"), tirx.BufferLoad(ub, [63, 1]), 64, 1)
    block = tirx.SBlock([], [], [], "SIMD_VF", tirx.Evaluate(pointer))
    with pytest.raises(ValueError, match="vector access exceeds"):
        _verify_vector_memory(block, [ub])


@pytest.mark.parametrize("container", ["body", "allocations", "inputs", "annotations"])
def test_fused_plan_cache_does_not_expose_mutable_containers(container, tmp_path):
    kernel = D.compile(allgather_gemm(1, 1), profile=D.P0B_PROFILE)
    original = kernel.plan()
    before = {r: tilelang.tvm.ir.save_json(f) for r, f in kernel._funcs().items()}
    gemm = next(item for item in original[0] if item[0] == "gemm")
    if container == "annotations":
        gemm[7]["unexpected"] = 1
    else:
        gemm[{"body": 3, "allocations": 4, "inputs": 5}[container]].clear()
    with pytest.raises(ValueError, match="validated current plan"):
        kernel.build(shmem_root=fake_headers(tmp_path), plans=original)
    restored = kernel.plan()
    restored_gemm = next(item for item in restored[0] if item[0] == "gemm")
    assert restored_gemm[3] and restored_gemm[4] and restored_gemm[5]
    assert "unexpected" not in restored_gemm[7]
    assert {r: tilelang.tvm.ir.save_json(f) for r, f in kernel._funcs().items()} == before


def test_fused_same_epoch_cache_preserves_interval_and_progress_checks(monkeypatch):
    kernel = D.compile(exchange(3, 1), profile=D.P0A_PROFILE)
    calls = []
    actual = kernel._plan_program

    def observed(*args, **kwargs):
        calls.append(kwargs["scalars"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(kernel, "_plan_program", observed)
    kernel.plan(epoch_base=3, epoch_end=6)
    count = len(calls)
    kernel.plan(epoch_base=3, epoch_end=6)
    assert len(calls) == count
    with pytest.raises(ValueError, match="outside the prepared"):
        kernel.plan(epoch_base=3, epoch_end=4)
    # Cached expansion cannot reuse a previous progress result: stale high
    # signals would make the new publications decrease.
    with pytest.raises(ValueError, match="must not decrease"):
        kernel.plan(epoch_base=3, epoch_end=6, initial_signals={(0, "Ready", 0): 100})


def test_fused_scalar_specializations_are_distinct_and_bounded():
    from tvm import tirx

    original = exchange(1, 1)
    choice = tirx.Var("choice", "int32")
    program = tirx.PrimFunc([*original.params, choice], original.body, buffer_map=original.buffer_map, attrs=original.attrs)
    kernel = D.compile(program, profile=D.P0A_PROFILE)
    for value in range(10):
        kernel.plan(scalars={"choice": value})
        kernel._funcs()
    assert len(kernel._reference_cache) == 4
    assert len(kernel._expanded_cache) <= 8
    assert len(kernel._description_cache) <= 128
    assert len({key for key in kernel._reference_cache}) == 4
    with pytest.raises(ValueError, match="scalar integer values"):
        kernel.plan(scalars={"choice": 1.5})


def test_fused_warm_build_revalidates_headers_and_helper(monkeypatch, tmp_path):
    root = fake_headers(tmp_path)
    kernel = D.compile(exchange(1, 1), profile=D.P0A_PROFILE, shmem_root=root)
    monkeypatch.setattr(tilelang, "compile", lambda *args, **kwargs: object())
    plans = kernel.plan()
    kernel.build(shmem_root=root, plans=plans)
    (root / "include/shmem.h").write_text("changed after cache warmup")
    with pytest.raises(ValueError, match="SHMEM device headers changed"):
        kernel.build(shmem_root=root, plans=plans)


@pytest.mark.parametrize("case", ["put", "get", "vector", "gemm"])
def test_fused_affine_epoch_proof_reuses_validated_expansion(monkeypatch, case):
    source = allgather_gemm(3, 1) if case == "gemm" else vector_exchange(3, 1) if case == "vector" else exchange(3, 1, get=case == "get")
    kernel = D.compile(source, profile=D.P0B_PROFILE if case == "gemm" else D.P0A_PROFILE)
    first = kernel.plan()
    assert kernel._active_reference["proof"] is not None
    monkeypatch.setattr(kernel, "_plan_program", lambda *args, **kwargs: pytest.fail("proven epoch translation replanned source"))
    second = kernel.plan(epoch_base=9, epoch_end=12)
    for rank in (0, 1):
        for before, after in zip(first[rank], second[rank]):
            if before[0] in ("signal", "wait_signal"):
                assert after[2] == before[2] + 9


def test_fused_unproven_epoch_control_falls_back_to_actual_expansion(monkeypatch):
    from tvm import tirx

    original = exchange(1, 1)
    epoch = next(param for param in original.params if param.name == "epoch0")
    source = original.with_body(tirx.IfThenElse(epoch >= 2, original.body, original.body))
    kernel = D.compile(source, profile=D.P0A_PROFILE)
    kernel.plan()
    assert kernel._active_reference["proof"] is None
    calls = []
    actual = kernel._plan_program

    def observed(*args, **kwargs):
        calls.append(kwargs["rank"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(kernel, "_plan_program", observed)
    kernel.plan(epoch_base=2, epoch_end=3)
    assert calls == [0, 1]


def test_fused_non_affine_epoch_hidden_at_zero_and_one_is_rejected():
    from tvm import tirx

    original = exchange(1, 1)
    epoch = next(param for param in original.params if param.name == "epoch0")
    body = tirx.stmt_functor.substitute(original.body, {epoch: epoch + epoch * (epoch - 1)})
    kernel = D.compile(original.with_body(body), profile=D.P0A_PROFILE)
    kernel.plan()
    assert kernel._active_reference["proof"] is None
    with pytest.raises(ValueError, match="epoch0 may only translate"):
        kernel.plan(epoch_base=2)


def test_fused_epoch_proof_does_not_hide_intermediate_overflow():
    from tvm import tirx
    from tilelang.distributed.abi import MAX_EPOCH

    original = exchange(1, 1)
    epoch = next(param for param in original.params if param.name == "epoch0")
    body = tirx.stmt_functor.substitute(original.body, {epoch: (epoch + 10) - 10})
    kernel = D.compile(original.with_body(body), profile=D.P0A_PROFILE)
    kernel.plan()
    assert kernel._active_reference["proof"][1] <= MAX_EPOCH - 10
    with pytest.raises(ValueError, match="integer overflow"):
        kernel.plan(epoch_base=MAX_EPOCH - 2, epoch_end=MAX_EPOCH)


def test_failed_plan_cannot_rebind_previous_build_specialization(monkeypatch, tmp_path):
    from tvm import tirx

    original = exchange(1, 1)
    choice = tirx.Var("choice", "int32")
    program = tirx.PrimFunc([*original.params, choice], original.body, buffer_map=original.buffer_map, attrs=original.attrs)
    kernel = D.compile(program, profile=D.P0A_PROFILE)
    first = kernel.plan(scalars={"choice": 0})
    active = kernel._active_reference
    funcs = kernel._funcs()
    with pytest.raises(ValueError, match="positive and fit int32"):
        kernel.plan(scalars={"choice": 1}, epoch_end=0)
    assert kernel._active_reference is active
    built = []
    monkeypatch.setattr(tilelang, "compile", lambda func, **kwargs: built.append(func) or object())
    kernel.build(shmem_root=fake_headers(tmp_path), plans=first)
    assert all(actual.same_as(funcs[rank]) for rank, actual in enumerate(built))
