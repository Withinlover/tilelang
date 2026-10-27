"""Bounded packed GEMM regions whose M/N/K loops stay on the device.

This is an explicit extraction contract, not a general loop recognizer. Each
axis contains 1..16 tiles of size 64. No communication is permitted inside the
region: its complete GM read/write footprints remain visible to TD planning.
"""

from tilelang import tvm
from tvm import tirx

from tilelang.analysis.distributed_checker import op_name, tile_region
from .segments import SegmentCompileError, _constant, _verify_compact_buffer, _verify_gemm_body


MARKER = "td.gemm"
MAX_TILES = 16


def _sequence(body):
    return list(body.seq) if isinstance(body, tirx.SeqStmt) else [body]


def _loop(node, name, *, marker=False):
    if not isinstance(node, tirx.For) or node.kind != tirx.ForKind.SERIAL or node.thread_binding is not None:
        raise SegmentCompileError(f"packed GEMM requires a plain serial {name} tile loop")
    annotations = dict(node.annotations)
    if marker and MARKER in annotations and _constant(annotations.pop(MARKER), "packed GEMM marker") != 1:
        raise SegmentCompileError("packed GEMM requires td.gemm=1")
    if annotations:
        raise SegmentCompileError("packed GEMM loops do not support scheduling or other annotations")
    size = _constant(node.extent, f"packed GEMM {name} extent")
    if _constant(node.min, f"packed GEMM {name} minimum") != 0 or not 1 <= size <= MAX_TILES:
        raise SegmentCompileError(f"packed GEMM {name} requires 1..{MAX_TILES} tiles starting at zero")
    if node.loop_var.dtype != "int32":
        raise SegmentCompileError("packed GEMM tile loops require int32 indices")
    return size


def _copy(statement):
    if not isinstance(statement, tirx.Evaluate) or op_name(statement.value) != "tl.tileop.ascend_copy":
        raise SegmentCompileError("packed GEMM requires explicit Ascend tile copies")
    return statement.value


def _selection(region, expected_indices, shape, name):
    if region is None or region.buffer.scope() != "global" or len(region.region) < 4:
        raise SegmentCompileError(f"packed GEMM {name} requires a global [..., tile, tile, 64, 64] buffer")
    if len(region.region) != len(region.buffer.shape):
        raise SegmentCompileError(f"packed GEMM {name} region rank must match its buffer rank")
    _verify_compact_buffer(region.buffer, f"packed GEMM {name}")
    analyzer = tvm.arith.Analyzer()
    selections = list(region.region)
    expected_extents = [1, 1, 64, 64]
    for item, expected in zip(selections[-4:], expected_extents):
        if _constant(item.extent, f"{name} tile extent") != expected:
            raise SegmentCompileError(f"packed GEMM {name} must copy one complete 64x64 tile")
    for item, expected in zip(selections[-4:], [*expected_indices, 0, 0]):
        if not analyzer.can_prove(item.min == expected):
            raise SegmentCompileError(f"packed GEMM {name} has unsupported tile indices")
    leading = []
    for item, extent in zip(selections[:-4], region.buffer.shape[:-4]):
        start = _constant(item.min, f"{name} leading index")
        if _constant(item.extent, f"{name} leading extent") != 1 or not 0 <= start < _constant(extent, f"{name} shape"):
            raise SegmentCompileError(f"packed GEMM {name} requires in-bounds singleton leading dimensions")
        leading.append(tvm.ir.Range.from_min_extent(start, 1))
    # Complete packed trailing dimensions keep every phase contiguous; partial
    # slabs and tails need a distinct, separately validated contract.
    if [_constant(x, f"{name} packed shape") for x in region.buffer.shape[-4:]] != list(shape):
        raise SegmentCompileError(f"packed GEMM {name} requires trailing packed shape {shape}")
    return tirx.BufferRegion(region.buffer, [*leading, *[tvm.ir.Range.from_min_extent(0, x) for x in shape]])


def parse_tiled_gemm(loop, allocations, annotations=None):
    """Validate one region and return its unmarked body and full GM effects."""
    m = loop
    mt = _loop(m, "M", marker=True)
    n = m.body
    nt = _loop(n, "N")
    statements = _sequence(n.body)
    if len(statements) != 2:
        raise SegmentCompileError("packed GEMM N body requires a K loop followed by one output copy")
    k, output_copy = statements
    kt = _loop(k, "K")
    if len({m.loop_var, n.loop_var, k.loop_var}) != 3:
        raise SegmentCompileError("packed GEMM requires three distinct loop indices")
    inner = _sequence(k.body)
    if len(inner) != 5:
        raise SegmentCompileError("packed GEMM K body requires four input copies and one GEMM")
    a_call, b_call = _copy(inner[0]), _copy(inner[1])
    c_call = _copy(output_copy)
    inputs = [
        _selection(tile_region(a_call.args[0]), [m.loop_var, k.loop_var], (mt, kt, 64, 64), "A"),
        _selection(tile_region(b_call.args[0]), [n.loop_var, k.loop_var], (nt, kt, 64, 64), "B"),
    ]
    output = _selection(tile_region(c_call.args[1]), [m.loop_var, n.loop_var], (mt, nt, 64, 64), "C")
    regions = [*inputs, output]
    for index, (region, dtype) in enumerate(zip(regions, ["bfloat16", "bfloat16", "float32"])):
        if region.buffer.dtype != dtype:
            raise SegmentCompileError(f"packed GEMM requires {dtype} for {('A', 'B', 'C')[index]}")
        if any(region.buffer.data.same_as(old.buffer.data) for old in regions[:index]):
            raise SegmentCompileError("packed GEMM inputs/output require distinct GM allocations")
    gemm = inner[4]
    if not isinstance(gemm, tirx.Evaluate) or op_name(gemm.value) != "tl.tileop.gemm" or len(gemm.value.args) != 13:
        raise SegmentCompileError("packed GEMM K body must finish with one standard T.gemm")
    # The reduction must initialize once per output tile, then preserve L0C.
    analyzer = tvm.arith.Analyzer()
    if not analyzer.can_prove(gemm.value.args[9] == (k.loop_var == 0)):
        raise SegmentCompileError("packed GEMM requires clear_accum=(k == 0) for complete K accumulation")
    accumulator_region = gemm.value.args[2]
    if op_name(accumulator_region) != "tl.region" or _constant(accumulator_region.args[1], "accumulator access") != 3:
        raise SegmentCompileError("packed GEMM accumulator must declare its read/write reduction access")

    canonical = [tirx.decl_buffer((64, 64), dtype, name=name) for name, dtype in zip("ABC", ["bfloat16", "bfloat16", "float32"])]
    known_calls = [a_call.args[0], b_call.args[0], c_call.args[1]]
    if any(
        op_name(call) != "tl.region" or not isinstance(call.args[0], tirx.BufferLoad) or call.args[0].predicate is not None
        for call in known_calls
    ):
        raise SegmentCompileError("packed GEMM requires explicit unpredicated GM tile regions")

    def tile(node):
        if isinstance(node, tirx.Call) and op_name(node) == "tl.region":
            if node.same_as(accumulator_region):
                return tirx.Call(node.dtype, node.op, [node.args[0], 2, *node.args[2:]], node.annotations)
            for original, replacement in zip(known_calls, canonical):
                if node.same_as(original):
                    return tirx.Call(node.dtype, node.op, [tirx.BufferLoad(replacement, [0, 0]), node.args[1], 64, 64], node.annotations)
        return None

    def first_accumulation(node):
        if isinstance(node, tirx.Call) and op_name(node) == "tl.tileop.gemm":
            args = list(node.args)
            args[9] = tirx.const(True, "bool")
            return tirx.Call(node.dtype, node.op, args, node.annotations)
        return None

    # Existing strict six-operation validator checks every local allocation,
    # scope, dtype, argument, transpose and annotation. Only the already-proven
    # clear expression is replaced: substituting loop indices here could hide
    # an unsafe k-dependent local pointer that happened to be zero at k=0.
    tile_body = tirx.stmt_functor.ir_transform(tirx.SeqStmt([*inner, output_copy]), tile, first_accumulation)
    used = _verify_gemm_body(tile_body, canonical)
    if len(allocations) != 5 or any(not any(b.same_as(x) for x in used) for b in allocations):
        raise SegmentCompileError("packed GEMM requires exactly its five independent local allocations")
    # Delegate layout verification as well, using an ordinary validated tile.
    from .segments import extract_segment

    simple_regions = [tirx.BufferRegion(b, [tvm.ir.Range(0, 64), tvm.ir.Range(0, 64)]) for b in canonical]
    extract_segment(_sequence(tile_body), allocations, simple_regions[:2], simple_regions[2], annotations=annotations)
    clean = tirx.For(m.loop_var, m.min, m.extent, m.kind, m.body, m.thread_binding, {})
    return clean, inputs, output


def extract_tiled_segment(loop, allocations, inputs, output, annotations=None):
    """Rebase a validated packed region without expanding its device loops."""
    clean, actual_inputs, actual_output = parse_tiled_gemm(loop, allocations, annotations)
    actual = [*actual_inputs, actual_output]
    supplied = [*inputs, output]
    if len(inputs) != 2 or any(not tvm.ir.structural_equal(x, y) for x, y in zip(actual, supplied)):
        raise SegmentCompileError("packed GEMM supplied regions differ from its complete read/write footprints")
    canonical = [
        tirx.decl_buffer(tuple(_constant(r.extent, "packed extent") for r in region.region[-4:]), region.buffer.dtype, name=name)
        for name, region in zip("ABC", actual)
    ]

    def rebase(node):
        if isinstance(node, tirx.Call) and op_name(node) == "tl.region":
            load = node.args[0]
            for region, replacement in zip(actual, canonical):
                if isinstance(load, tirx.BufferLoad) and load.buffer.same_as(region.buffer):
                    return tirx.Call(
                        node.dtype,
                        node.op,
                        [tirx.BufferLoad(replacement, list(load.indices[-4:])), node.args[1], *node.args[-4:]],
                        node.annotations,
                    )
        return None

    body = tirx.stmt_functor.ir_transform(clean, rebase, None)
    attrs = dict(annotations or {})
    if "layout_map" not in attrs:
        from tilelang.layout import make_ascend_l0c_layout

        acc = next(x for x in allocations if x.scope() == "shared.l0c")
        attrs["layout_map"] = {acc.data: make_ascend_l0c_layout(acc)}
    block = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=allocations, annotations=attrs)
    bx = tirx.Var("bx", "int32")
    axis = tirx.IterVar(tvm.ir.Range(0, 1), bx, tirx.IterVar.ThreadIndex, "blockIdx.x")
    body = tirx.For(bx, 0, 1, tirx.ForKind.THREAD_BINDING, tirx.SBlockRealize([], True, block), thread_binding=axis)
    body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "root", body))
    return tirx.PrimFunc([b.data for b in canonical], body, buffer_map={b.data: b for b in canonical}).with_attr(
        {"global_symbol": "p0_packed_gemm_segment", "tir.noalias": True}
    )
