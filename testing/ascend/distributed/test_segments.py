"""Local-only contracts for extraction of computation from distributed TIR."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.distributed.segments import SegmentCompileError, adapt_cann91_gemm64, extract_segment
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _fixture(*, selected=1, actual=1, transpose=True):
    @T.prim_func
    def main(
        A: T.Tensor((2, 64, 64), "bfloat16"),
        B: T.Tensor((64, 64), "bfloat16"),
        C: T.Tensor((2, 64, 64), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 64), "bfloat16")
            b_l1 = T.alloc_l1((64, 64), "bfloat16")
            a_l0 = T.alloc_l0a((64, 64), "bfloat16")
            b_l0 = T.alloc_l0b((64, 64), "bfloat16")
            acc = T.alloc_l0c((64, 64), "float32")
            T.copy(A[actual, :, :], a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0, transpose=transpose)
            T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C[actual, :, :])

    blocks = []
    post_order_visit(main.body, lambda n: blocks.append(n) if isinstance(n, tirx.SBlock) and n.name_hint == "tilelang_root" else None)
    block = blocks[0]
    a, b, c = main.buffer_map.values()
    r = tilelang.tvm.ir.Range.from_min_extent
    regions = [
        tirx.BufferRegion(a, [r(selected, 1), r(0, 64), r(0, 64)]),
        tirx.BufferRegion(b, [r(0, 64), r(0, 64)]),
        tirx.BufferRegion(c, [r(selected, 1), r(0, 64), r(0, 64)]),
    ]
    return list(block.body.seq), list(block.alloc_buffers), regions, block.annotations


def _extract(fixture):
    statements, allocations, regions, annotations = fixture
    return extract_segment(statements, allocations, regions[:2], regions[2], annotations=annotations)


def test_selected_rank_slot_is_rebased_and_original_operations_are_lowered():
    statements, allocations, regions, annotations = _fixture()
    result = extract_segment(statements, allocations, regions[:2], regions[2], annotations=annotations)
    assert [[int(n) for n in b.shape] for b in result.buffer_map.values()] == [[64, 64]] * 3
    calls = []
    post_order_visit(result.body, lambda n: calls.append(n) if isinstance(n, tirx.Call) else None)
    assert sum(isinstance(c.op, tilelang.tvm.ir.Op) and c.op.name == "tl.tileop.gemm" for c in calls) == 1
    assert sum(isinstance(c.op, tilelang.tvm.ir.Op) and c.op.name == "tl.tileop.ascend_copy" for c in calls) == 5
    # Lower the extracted TIR rather than replacing its computation with a
    # separately written GEMM. The B transpose remains visible in device code.
    source = tilelang.lower(result, target="ascend").kernel_source
    assert "asc_copy_l12l0b_transpose" in source
    assert "__global__ __cube__" in source
    assert "asc_mmad" in source
    compatible = adapt_cann91_gemm64(source)
    assert "asc_copy_l12l0b_transpose" in compatible
    assert "asc_set_l0c2gm_nz2nd(1, 0, 0)" in compatible
    assert "(64ULL << 32) | (1ULL << 16) | 1ULL" in compatible


@pytest.mark.parametrize("selected,actual", [(0, 1), (2, 2)])
def test_other_slot_or_out_of_bounds_selection_is_rejected(selected, actual):
    with pytest.raises(SegmentCompileError, match="another GM tile|in bounds"):
        _extract(_fixture(selected=selected, actual=actual))


def test_nonstandard_gemm_math_is_rejected_before_using_helper_reference():
    with pytest.raises(SegmentCompileError, match="requires the B L1-to-L0 transpose"):
        _extract(_fixture(transpose=False))


def test_duplicate_or_missing_gemm_is_rejected():
    statements, allocations, regions, annotations = _fixture()
    statements.insert(5, statements[4])
    with pytest.raises(SegmentCompileError, match="exactly one original T.gemm|four input copies"):
        extract_segment(statements, allocations, regions[:2], regions[2], annotations=annotations)


def test_compatibility_profile_rejects_unrecognized_source():
    with pytest.raises(SegmentCompileError, match="preamble"):
        adapt_cann91_gemm64("extern int unrelated();")


def test_sdk_compatibility_is_explicit_and_part_of_binary_cache(monkeypatch, tmp_path):
    from tilelang.ascend.backend import tilelang_callback_ascend_compile
    from tilelang.cache.ascend_binary_cache import AscendBinaryCache
    from tilelang.contrib import bisheng
    from tilelang.env import env
    from tilelang.transform import PassConfigKey
    from tvm.target import Target

    source = tilelang.lower(_extract(_fixture()), target={"kind": "ascend", "arch": "dav-3510"}).kernel_source
    calls = []
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    monkeypatch.setattr(bisheng, "compile_ascend", lambda code, **kwargs: calls.append(code) or bytearray(b"compiled"))
    monkeypatch.setattr(AscendBinaryCache, "_get_tilelang_lib_stamp", lambda: None)
    target = Target({"kind": "ascend", "arch": "dav-3510"})
    config = {PassConfigKey.TL_ASCEND_SDK_COMPATIBILITY: "cann-9.1-gemm64"}
    tilelang_callback_ascend_compile(source, target)
    tilelang_callback_ascend_compile(source, target, config)
    tilelang_callback_ascend_compile(source, target, config)
    assert calls == [source, adapt_cann91_gemm64(source)]
    with pytest.raises(ValueError, match="Unknown Ascend SDK compatibility"):
        tilelang_callback_ascend_compile(source, target, {PassConfigKey.TL_ASCEND_SDK_COMPATIBILITY: "unknown"})
    with pytest.raises(ValueError, match="requires dav-3510"):
        tilelang_callback_ascend_compile(source, Target({"kind": "ascend", "arch": "dav-2201"}), config)


def _replace_call(fixture, index, *, args=None, annotations=None):
    statements, allocations, regions, block_annotations = fixture
    original = statements[index].value
    statements[index] = tirx.Evaluate(
        tirx.Call(
            original.dtype,
            original.op,
            original.args if args is None else args,
            original.annotations if annotations is None else annotations,
            original.span,
        )
    )
    return statements, allocations, regions, block_annotations


@pytest.mark.parametrize(
    "index,annotation,value",
    [(2, "l0_pitch", 8), (5, "unit_flag_ctrl", 1), (3, "sub_blockid", 1), (4, "unit_flag_ctrl", 1)],
)
def test_layout_and_sync_operation_annotations_cannot_be_ignored(index, annotation, value):
    fixture = _fixture()
    annotations = {**dict(fixture[0][index].value.annotations), annotation: tirx.IntImm("int32", value)}
    with pytest.raises(SegmentCompileError, match="Unsupported P0.*annotations"):
        _extract(_replace_call(fixture, index, annotations=annotations))


@pytest.mark.parametrize("mutation", ["extra-argument", "read-write-mask", "region-annotation", "predicate"])
def test_noncanonical_copy_signature_is_rejected(mutation):
    fixture = _fixture()
    original = fixture[0][2].value
    args = list(original.args)
    if mutation == "extra-argument":
        args.append(tirx.IntImm("int32", 0))
        diagnostic = "requires 2 arguments"
    else:
        region = args[0]
        region_args = list(region.args)
        annotations = dict(region.annotations)
        if mutation == "read-write-mask":
            region_args[1] = tirx.IntImm("int32", 3)
            diagnostic = "requires access mask 1"
        elif mutation == "region-annotation":
            annotations["layout"] = "unvalidated"
            diagnostic = "no annotations"
        else:
            load = region_args[0]
            region_args[0] = tirx.BufferLoad(load.buffer, load.indices, predicate=tirx.const(False, "bool"))
            diagnostic = "unpredicated"
        args[0] = tirx.Call(region.dtype, region.op, region_args, annotations)
    with pytest.raises(SegmentCompileError, match=diagnostic):
        _extract(_replace_call(fixture, 2, args=args))


def _replace_buffer(fixture, original, replacement):
    from tvm.tirx.stmt_functor import ir_transform

    statements, allocations, regions, annotations = fixture

    def replace(node):
        if isinstance(node, tirx.BufferLoad) and node.buffer.same_as(original):
            return tirx.BufferLoad(replacement, node.indices, node.predicate, node.span)
        return None

    statements = [ir_transform(statement, replace, None) for statement in statements]
    allocations = [replacement if buffer.same_as(original) else buffer for buffer in allocations]
    return statements, allocations, regions, annotations


def test_two_gemm_inputs_cannot_overwrite_the_same_l1_allocation():
    fixture = _fixture()
    a_l1, b_l1 = fixture[1][:2]
    with pytest.raises(SegmentCompileError, match="five independent local buffer allocations"):
        _extract(_replace_buffer(fixture, b_l1, a_l1))


@pytest.mark.parametrize("mutation", ["stride", "offset", "shape"])
def test_noncompact_local_storage_cannot_use_compact_cpu_gemm(mutation):
    fixture = _fixture()
    original = fixture[1][0]
    replacement = tirx.decl_buffer(
        (64, 80) if mutation == "shape" else (64, 64),
        original.dtype,
        name=original.name,
        data=original.data,
        strides=(80, 1) if mutation == "stride" else None,
        elem_offset=1 if mutation == "offset" else 0,
        scope=original.scope(),
    )
    with pytest.raises(SegmentCompileError, match="compact row-major|zero element offset|requires shape"):
        _extract(_replace_buffer(fixture, original, replacement))


def test_custom_accumulator_layout_is_not_silently_replaced_by_default_gemm():
    from tilelang.layout import make_ascend_major_k_layout

    statements, allocations, regions, annotations = _fixture()
    accumulator = allocations[-1]
    annotations = {"layout_map": {accumulator.data: make_ascend_major_k_layout(accumulator)}}
    with pytest.raises(SegmentCompileError, match="default L0C layout"):
        extract_segment(statements, allocations, regions[:2], regions[2], annotations=annotations)


def test_noncompact_global_storage_cannot_be_rebased_as_contiguous_inputs():
    statements, allocations, regions, annotations = _fixture()
    original = regions[0].buffer
    strided = tirx.decl_buffer(original.shape, original.dtype, data=original.data, strides=(8192, 128, 1))
    regions[0] = tirx.BufferRegion(strided, regions[0].region)
    with pytest.raises(SegmentCompileError, match="global buffer A requires compact row-major"):
        extract_segment(statements, allocations, regions[:2], regions[2], annotations=annotations)


@pytest.mark.parametrize("index,value", [(3, True), (7, 32), (9, False)])
def test_transpose_reduction_extent_and_accumulator_reuse_must_match_oracle(index, value):
    fixture = _fixture()
    args = list(fixture[0][4].value.args)
    args[index] = tirx.const(value, args[index].dtype)
    with pytest.raises(SegmentCompileError, match="standard 64x64 BF16 A @ B with cleared FP32 accumulation"):
        _extract(_replace_call(fixture, 4, args=args))
