"""Finite packed GEMM extraction keeps real M/N/K work in device loops."""

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.language.distributed as TD
from tilelang import tvm
from tvm import tirx
from tilelang.ascend.distributed.plan import plan_program
from tilelang.ascend.distributed.segments import SegmentCompileError, compile_segment, extract_segment


G = TD.GroupSpec("world", 2)


def packed_program(mt=2, nt=2, kt=2, *, clear="first", bad_local=False):
    @T.prim_func
    def program(
        A: TD.SymBuffer((mt, kt, 64, 64), "bfloat16", group=G, remote_access="read"),
        B: TD.SymBuffer((nt, kt, 64, 64), "bfloat16", group=G, remote_access="read"),
        C: TD.SymBuffer((mt, nt, 64, 64), "float32", group=G, remote_access="write"),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 64), "bfloat16")
            b_l1 = T.alloc_l1((64, 64), "bfloat16")
            a_l0 = T.alloc_l0a((64, 64), "bfloat16")
            b_l0 = T.alloc_l0b((64, 64), "bfloat16")
            c_l0 = T.alloc_l0c((64, 64), "float32")
            for m in T.serial(mt, annotations={"td.gemm": 1}):
                for n in T.serial(nt):
                    for k in T.serial(kt):
                        T.copy(A[m, k, :, :], a_l1)
                        T.copy(B[n, k, :, :], b_l1)
                        if bad_local:
                            T.copy(a_l1[k : k + 64, :], a_l0)
                        else:
                            T.copy(a_l1, a_l0)
                        T.copy(b_l1, b_l0, transpose=True)
                        if clear == "always":
                            T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=True)
                        elif clear == "never":
                            T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=False)
                        else:
                            T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=(k == 0))
                    T.copy(c_l0, C[m, n, :, :])

    return program


def get_segment(program):
    plan, _ = plan_program(program, rank=0, scalars={"epoch0": 0}, world_size=2)
    gemms = [item for item in plan if item[0] == "gemm"]
    assert len(gemms) == 1
    return gemms[0]


@pytest.mark.parametrize("sizes", [(1, 1, 1), (2, 3, 2), (16, 16, 16)])
def test_packed_plan_retains_three_device_loops_and_whole_regions(sizes):
    mt, nt, kt = sizes
    item = get_segment(packed_program(*sizes))
    assert item[1][0].count == mt * kt * 4096
    assert item[1][1].count == nt * kt * 4096
    assert item[2].count == mt * nt * 4096
    assert len(item[3]) == 1
    segment = extract_segment(item[3], item[4], item[5], item[6], annotations=item[7])
    loops, calls = [], []
    tirx.stmt_functor.post_order_visit(segment.body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    tirx.stmt_functor.post_order_visit(segment.body, lambda node: calls.append(node) if isinstance(node, tirx.Call) else None)
    assert sorted(int(loop.extent) for loop in loops if loop.thread_binding is None) == sorted(sizes)
    gemm = [call for call in calls if isinstance(call.op, tvm.ir.Op) and call.op.name == "tl.tileop.gemm"]
    assert len(gemm) == 1
    assert isinstance(gemm[0].args[9], tirx.EQ)
    assert all("td.gemm" not in loop.annotations for loop in loops)


@pytest.mark.parametrize("clear", ["always", "never"])
def test_packed_k_accumulator_must_clear_only_first_iteration(clear):
    with pytest.raises(SegmentCompileError, match="clear_accum"):
        get_segment(packed_program(clear=clear))


def test_packed_local_pointer_cannot_hide_k_dependence():
    with pytest.raises(SegmentCompileError, match="must be constant|tile origin"):
        get_segment(packed_program(bad_local=True))


@pytest.mark.parametrize("sizes", [(17, 1, 1), (1, 17, 1), (1, 1, 17)])
def test_packed_capacity_diagnostic(sizes):
    with pytest.raises(SegmentCompileError, match="1..16 tiles"):
        get_segment(packed_program(*sizes))


def test_packed_extraction_rejects_incomplete_footprint():
    item = get_segment(packed_program())
    inputs = list(item[5])
    region = inputs[0]
    bounds = list(region.region)
    bounds[0] = tvm.ir.Range(0, 1)
    inputs[0] = tirx.BufferRegion(region.buffer, bounds)
    with pytest.raises(SegmentCompileError, match="complete read/write footprints"):
        extract_segment(item[3], item[4], inputs, item[6], annotations=item[7])


def test_packed_legacy_simulator_segment_rejects_wrong_abi(tmp_path):
    item = get_segment(packed_program())
    with pytest.raises(SegmentCompileError, match="simulator segment ABI supports GEMM64 only"):
        compile_segment(item[3], item[4], item[5], item[6], tmp_path, annotations=item[7])


def test_packed_read_indices_must_cover_the_declared_reduction():
    program = packed_program()

    def wrong_k(node):
        if isinstance(node, tirx.BufferLoad) and node.buffer.name == "A":
            return tirx.BufferLoad(node.buffer, [node.indices[0], 0, *node.indices[2:]])
        return None

    program = program.with_body(tirx.stmt_functor.ir_transform(program.body, wrong_k, None))
    with pytest.raises(SegmentCompileError, match="unsupported tile indices"):
        get_segment(program)


def test_packed_segment_lower_keeps_dynamic_accumulation():
    item = get_segment(packed_program())
    segment = extract_segment(item[3], item[4], item[5], item[6], annotations=item[7])
    source = tilelang.lower(segment, target="ascend").kernel_source
    assert source.count("asc_mmad(") == 1
    assert "== 0)" in source
    assert source.count("for (") >= 3
