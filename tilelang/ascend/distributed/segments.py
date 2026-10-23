"""Extract local GEMM computation from a distributed TIR program.

This module compiles the supplied statements, including their original copies,
GEMM, and on-chip buffers. The explicit CANN 9.1 profile only translates the
small fixed GEMM's changed SDK API spellings; it does not supply another kernel.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

import tilelang
from tilelang import tvm
from tilelang.contrib import bisheng
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit


class SegmentCompileError(ValueError):
    """A computation fragment cannot be safely extracted or compiled."""


def _constant(expr, description):
    value = tvm.arith.Analyzer().simplify(expr)
    if not isinstance(value, tirx.IntImm):
        raise SegmentCompileError(f"{description} must be constant, got {value}")
    return int(value)


def _verify_compact_buffer(buffer, description, *, shape=None):
    extents = [_constant(value, f"{description} shape") for value in buffer.shape]
    if any(extent <= 0 for extent in extents) or (shape is not None and extents != list(shape)):
        raise SegmentCompileError(f"{description} requires shape {shape}, got {extents}")
    if _constant(buffer.elem_offset, f"{description} element offset") != 0 or buffer.axis_separators:
        raise SegmentCompileError(f"{description} requires zero element offset and no axis separators")
    if buffer.strides:
        expected, pitch = [], 1
        for extent in reversed(extents):
            expected.insert(0, pitch)
            pitch *= extent
        actual = [_constant(value, f"{description} stride") for value in buffer.strides]
        if actual != expected:
            raise SegmentCompileError(f"{description} requires compact row-major strides {expected}, got {actual}")


def _verify_gemm_body(body, canonical):
    """Check that helper's A @ B reference matches the supplied computation."""
    sequence = list(body.seq) if isinstance(body, tirx.SeqStmt) else [body]
    expected = ["tl.tileop.ascend_copy"] * 4 + ["tl.tileop.gemm", "tl.tileop.ascend_copy"]
    if len(sequence) != len(expected):
        raise SegmentCompileError("P0 segments require four input copies, one GEMM, and one output copy")
    ops = []
    for statement, name in zip(sequence, expected):
        if not isinstance(statement, tirx.Evaluate) or not isinstance(statement.value, tirx.Call):
            raise SegmentCompileError("P0 GEMM fragments must contain direct tile operations")
        call = statement.value
        if not isinstance(call.op, tvm.ir.Op) or call.op.name != name:
            raise SegmentCompileError("P0 GEMM fragment copy/GEMM order is unsupported")
        expected_arguments = 13 if name == "tl.tileop.gemm" else 2
        if call.dtype != "handle" or len(call.args) != expected_arguments:
            raise SegmentCompileError(f"P0 {name} requires {expected_arguments} arguments and a handle result")
        allowed_annotations = set() if name == "tl.tileop.gemm" else {"transpose"}
        unsupported = set(call.annotations) - allowed_annotations
        if unsupported:
            raise SegmentCompileError(f"Unsupported P0 {name} annotations: {sorted(unsupported)}")
        ops.append(call)

    def region_buffer(region, mask):
        if not isinstance(region, tirx.Call) or region.op.name != "tl.region":
            raise SegmentCompileError("Expected an explicit tile region")
        if region.dtype != "handle" or len(region.args) != 4 or region.annotations:
            raise SegmentCompileError("P0 tile regions require exactly two dimensions and no annotations")
        if _constant(region.args[1], "region access mask") != mask:
            raise SegmentCompileError(f"P0 tile region requires access mask {mask}")
        load = region.args[0]
        if not isinstance(load, tirx.BufferLoad) or load.predicate is not None or len(load.indices) != 2:
            raise SegmentCompileError("P0 tile regions require an unpredicated two-dimensional buffer access")
        if any(_constant(i, "tile origin") != 0 for i in load.indices) or [_constant(e, "tile extent") for e in region.args[2:]] != [
            64,
            64,
        ]:
            raise SegmentCompileError("P0 local GEMM operations must cover the full 64x64 tile")
        return load.buffer

    edges = [(region_buffer(op.args[0], 1), region_buffer(op.args[1], 2)) for op in [*ops[:4], ops[5]]]
    a_l1, b_l1, a_l0, b_l0 = (edges[i][1] for i in range(4))
    acc = region_buffer(ops[4].args[2], 2)
    required_edges = [(canonical[0], a_l1), (canonical[1], b_l1), (a_l1, a_l0), (b_l1, b_l0), (acc, canonical[2])]
    if any(not actual.same_as(wanted) for edge, expected_edge in zip(edges, required_edges) for actual, wanted in zip(edge, expected_edge)):
        raise SegmentCompileError("The fragment does not compute the supplied A and B regions into C")
    if not region_buffer(ops[4].args[0], 1).same_as(a_l0) or not region_buffer(ops[4].args[1], 1).same_as(b_l0):
        raise SegmentCompileError("GEMM must consume the copied A and B tiles")
    local_buffers = [a_l1, b_l1, a_l0, b_l0, acc]
    for index, buffer in enumerate(local_buffers):
        if any(buffer.data.same_as(other.data) for other in local_buffers[:index]):
            raise SegmentCompileError("P0 GEMM requires five independent local buffer allocations")
        _verify_compact_buffer(buffer, f"local buffer {buffer.name}", shape=(64, 64))
    for buffer, scope, dtype in [
        (a_l1, "shared.l1", "bfloat16"),
        (b_l1, "shared.l1", "bfloat16"),
        (a_l0, "shared.l0a", "bfloat16"),
        (b_l0, "shared.l0b", "bfloat16"),
        (acc, "shared.l0c", "float32"),
    ]:
        if buffer.scope() != scope or buffer.dtype != dtype:
            raise SegmentCompileError(f"Expected {dtype} storage in {scope}")
    for index in (0, 1, 2, 5):
        if _constant(ops[index].annotations.get("transpose", 0), "copy transpose"):
            raise SegmentCompileError("Only the B L1-to-L0 copy may transpose")
    if _constant(ops[3].annotations.get("transpose", 0), "copy transpose") != 1:
        raise SegmentCompileError("Standard A @ B requires the B L1-to-L0 transpose")
    args = ops[4].args
    if [_constant(a, "GEMM argument") for a in args[3:]] != [0, 1, 64, 64, 64, 0, 1, 0, 0, 0]:
        raise SegmentCompileError("P0 GEMM requires standard 64x64 BF16 A @ B with cleared FP32 accumulation")
    return local_buffers


def extract_segment(statements, alloc_buffers, input_regions, output_region, *, annotations=None):
    """Rebase three GM regions and retain the actual local computation body.

    The first two regions are A and B; the third is C. Leading dimensions must
    be singleton slices. A and B have BF16 storage and C has FP32 storage, each
    with a 64x64 trailing region. This matches cannsim-helper's current GEMM ABI.
    Runtime scalar substitutions must already have been applied by the caller.
    """
    if len(input_regions) != 2 or not statements:
        raise SegmentCompileError("GEMM segments need two inputs, one output, and a nonempty body")
    regions = [*input_regions, output_region]
    canonical = []
    bindings = []
    for name, region, dtype in zip(("A", "B", "C"), regions, ("bfloat16", "bfloat16", "float32")):
        if not isinstance(region, tirx.BufferRegion):
            raise SegmentCompileError(f"{name} must be a BufferRegion")
        shape = [_constant(r.extent, f"{name} extent") for r in region.region]
        if len(shape) < 2 or shape[-2:] != [64, 64] or any(n != 1 for n in shape[:-2]):
            raise SegmentCompileError(f"{name} must select one 64x64 tile, got {shape}")
        if region.buffer.dtype != dtype or region.buffer.scope() != "global":
            raise SegmentCompileError(f"{name} must have global {dtype} storage")
        if len(region.region) != len(region.buffer.shape):
            raise SegmentCompileError(f"{name} region rank differs from its buffer rank")
        _verify_compact_buffer(region.buffer, f"global buffer {name}")
        if any(region.buffer.data.same_as(old.buffer.data) for old in regions[: len(canonical)]):
            raise SegmentCompileError("GEMM inputs and output must have distinct GM allocations")
        for extent, selection in zip(region.buffer.shape, region.region):
            analyzer = tvm.arith.Analyzer()
            if not analyzer.can_prove(selection.min >= 0) or not analyzer.can_prove(selection.min + selection.extent <= extent):
                raise SegmentCompileError(f"{name} selected tile cannot be proven in bounds")
        buffer = tirx.decl_buffer((64, 64), dtype, name=name, strides=(64, 1))
        canonical.append(buffer)
        bindings.append((region.buffer, region, buffer))

    def binding(buffer):
        for original, region, replacement in bindings:
            if original.data.same_as(buffer.data):
                if not original.same_as(buffer):
                    raise SegmentCompileError("Aliased GM buffer views need explicit canonicalization before extraction")
                return region, replacement
        return None

    def rebase_load(load):
        found = binding(load.buffer)
        if found is None:
            if load.buffer.scope() == "global":
                raise SegmentCompileError(f"Unbound global buffer {load.buffer.name} in computation fragment")
            return load
        region, replacement = found
        analyzer = tvm.arith.Analyzer()
        indices = [analyzer.simplify(i - r.min) for i, r in zip(load.indices, region.region)]
        if any(not analyzer.can_prove(i == 0) for i in indices[:-2]):
            raise SegmentCompileError("A computation fragment accesses another GM tile")
        return tirx.BufferLoad(replacement, indices[-2:], load.predicate, load.span)

    def before(node):
        # Handle region calls before their BufferLoad children so leading unit
        # dimensions disappear from both the pointer and the region extents.
        if isinstance(node, tirx.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.region":
            load = node.args[0]
            if isinstance(load, tirx.BufferLoad) and binding(load.buffer) is not None:
                rebased = rebase_load(load)
                extents = list(node.args[2:])
                if any(_constant(e, "leading region extent") != 1 for e in extents[:-2]):
                    raise SegmentCompileError("A computation fragment accesses multiple GM tiles")
                analyzer = tvm.arith.Analyzer()
                for index, extent in zip(rebased.indices, extents[-2:]):
                    if not analyzer.can_prove(index >= 0) or not analyzer.can_prove(index + extent <= 64):
                        raise SegmentCompileError("A computation fragment accesses outside the selected GM tile")
                return tirx.Call(node.dtype, node.op, [rebased, node.args[1], *extents[-2:]], node.annotations, node.span)
        if isinstance(node, tirx.BufferLoad):
            return rebase_load(node)
        if isinstance(node, tirx.BufferStore) and node.buffer.scope() == "global":
            raise SegmentCompileError("GEMM segments require explicit T.copy for global writes")
        return None

    body = statements[0] if len(statements) == 1 else tirx.SeqStmt(statements)
    try:
        body = ir_transform(body, before, None)
    except RuntimeError as error:
        # TVM converts unregistered Python callback exception types to
        # RuntimeError when crossing its FFI boundary. Restore this API type.
        raise SegmentCompileError(str(error)) from error
    calls = []
    post_order_visit(body, lambda n: calls.append(n) if isinstance(n, tirx.Call) else None)
    gemms = [c for c in calls if isinstance(c.op, tvm.ir.Op) and c.op.name == "tl.tileop.gemm"]
    if len(gemms) != 1:
        raise SegmentCompileError(f"Expected exactly one original T.gemm, found {len(gemms)}")
    for call in calls:
        name = call.op.name if isinstance(call.op, tvm.ir.Op) else str(call.op)
        if name not in {"tl.region", "tl.tileop.ascend_copy", "tl.tileop.gemm"}:
            raise SegmentCompileError(f"Unsupported operation in P0 GEMM fragment: {name}")
    used_buffers = _verify_gemm_body(body, canonical)
    local_buffers = list(alloc_buffers)
    if len(local_buffers) != 5 or any(not any(buffer.same_as(used) for used in used_buffers) for buffer in local_buffers):
        raise SegmentCompileError("P0 computation fragment must declare exactly its five local buffers")
    if len({buffer.data for buffer in local_buffers}) != 5:
        raise SegmentCompileError("P0 computation fragment has duplicate local allocations")
    block_annotations = dict(annotations or {})
    if set(block_annotations) - {"layout_map"}:
        raise SegmentCompileError("P0 computation fragments only support the default L0C layout annotation")
    from tilelang.layout import make_ascend_l0c_layout

    accumulator = used_buffers[-1]
    default_layout = make_ascend_l0c_layout(accumulator)
    if "layout_map" in block_annotations:
        layouts = block_annotations["layout_map"]
        if len(layouts) != 1:
            raise SegmentCompileError("P0 computation fragments only support the default L0C layout")
        key, layout = next(iter(layouts.items()))
        data = key.data if isinstance(key, tirx.Buffer) else key
        if (
            not isinstance(data, tirx.Var)
            or not data.same_as(accumulator.data)
            or not tvm.ir.structural_equal(layout, default_layout, map_free_vars=True)
        ):
            raise SegmentCompileError("P0 computation fragments only support the default L0C layout")
    else:
        block_annotations["layout_map"] = {accumulator.data: default_layout}
    block = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=local_buffers, annotations=block_annotations)
    bx = tirx.Var("bx", "int32")
    axis = tirx.IterVar(tvm.ir.Range(0, 1), bx, tirx.IterVar.ThreadIndex, "blockIdx.x")
    body = tirx.For(bx, 0, 1, tirx.ForKind.THREAD_BINDING, tirx.SBlockRealize([], True, block), thread_binding=axis)
    body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "root", body))
    return tirx.PrimFunc([buffer.data for buffer in canonical], body, buffer_map={buffer.data: buffer for buffer in canonical}).with_attr(
        {"global_symbol": "p0_gemm_segment", "tir.noalias": True}
    )


def adapt_cann91_gemm64(source):
    """Translate only the audited 64x64 Cube GEMM SDK surface.

    Fail closed on an unfamiliar generated body. Packing of MTE2_NZ_PARA and
    Fixpipe defaults follows the installed CANN 9.1 public headers and their
    dav-3510 implementations. All copies, offsets and synchronization survive.
    """
    includes = "#include <tl_templates/ascend/common.h>\n#include <tl_templates/ascend/debug.h>\n"
    if source.count(includes) != 1 or "__global__ __cube__" not in source:
        raise SegmentCompileError("cann-9.1-gemm64 requires the standard pure Cube source preamble")
    # This profile does not compile arbitrary SIMD/debug template contents,
    # which target newer SDK APIs even when unused by this fragment.
    result = source.replace(includes, "#include <c_api/asc_simd.h>\n")
    replacements = [
        ("asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0)", "asc_set_gm2l1_nz_para((64ULL << 32) | (1ULL << 16) | 1ULL)", 2),
        ("static_cast<asc_load_l2_cache_mode>(0)", "0", 2),
        ("asc_set_l0c_copy_nz_para(1, 0, 0)", "asc_set_l0c2gm_nz2nd(1, 0, 0)", 1),
        (
            "static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0",
            "0, 0, 0, 0, 0, false, true, 0, 0, false, 0, false, false, false, false",
            1,
        ),
    ]
    for old, new, expected in replacements:
        if result.count(old) != expected:
            raise SegmentCompileError(f"cann-9.1-gemm64 expected {expected} occurrences of {old!r}")
        result = result.replace(old, new)
    allowed = {
        "asc_init",
        "asc_set_gm2l1_nz_para",
        "asc_copy_gm2l1_nd2nz",
        "asc_sync_notify",
        "asc_sync_wait",
        "asc_copy_l12l0a",
        "asc_copy_l12l0b_transpose",
        "asc_mmad",
        "asc_set_l0c2gm_nz2nd",
        "asc_copy_l0c2gm",
    }
    unknown = set(re.findall(r"\b(asc_\w+)\s*\(", result)) - allowed
    if unknown:
        raise SegmentCompileError(f"cann-9.1-gemm64 does not translate these SDK operations: {sorted(unknown)}")
    return result


def compile_segment(statements, alloc_buffers, input_regions, output_region, output_dir, *, profile="native", annotations=None):
    """Extract and compile a local fragment, returning artifact/ABI metadata.

    ``profile='cann-9.1-gemm64'`` must be selected explicitly for the locally
    installed older Simulator SDK. No implicit fallback or precompiled binary
    is used. The resulting executable CCE ELF is accepted by cannsim-helper.
    """
    if profile not in {"native", "cann-9.1-gemm64"}:
        raise SegmentCompileError(f"Unknown compilation profile {profile!r}")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    func = extract_segment(statements, alloc_buffers, input_regions, output_region, annotations=annotations)
    (out / "segment.tir").write_text(func.script(show_meta=True))
    (out / "segment.ir.json").write_text(tvm.ir.save_json(func))
    source = tilelang.lower(func, target="ascend").kernel_source
    (out / "generated.asc").write_text(source)
    compiled_source = adapt_cann91_gemm64(source) if profile == "cann-9.1-gemm64" else source
    source_path = out / "compiled.asc"
    source_path.write_text(compiled_source)
    (out / "sdk_compatibility.diff").write_text(
        "".join(
            difflib.unified_diff(source.splitlines(True), compiled_source.splitlines(True), fromfile="generated.asc", tofile="compiled.asc")
        )
    )
    options = ["-I" + tilelang.env.TILELANG_TEMPLATE_PATH]
    object_path = out / "segment.aibin"
    commands = [
        [
            bisheng.find_bisheng_path(),
            *bisheng.get_bisheng_compile_options("dav-3510", options),
            "--cce-aicore-only",
            str(source_path),
            "-o",
            str(out / "segment.rel.o"),
        ],
        [bisheng.find_ld_lld_path(), *bisheng.get_aibin_linker_options(), str(out / "segment.rel.o"), "-o", str(object_path)],
    ]
    (out / "commands.json").write_text(json.dumps(commands, indent=2))
    with (out / "compile.log").open("w") as log:
        for command in commands:
            log.write("$ " + shlex.join(command) + "\n")
            log.flush()
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=120)
            if process.returncode:
                raise SegmentCompileError(f"Bisheng compilation failed ({process.returncode}); see {out / 'compile.log'}")
    metadata = {
        "profile": profile,
        "target": "ascend",
        "arch": "dav-3510",
        "kernel_type": "cube",
        "expected_cores": ["AIC"],
        "kernel_extracted_from_tir": True,
        "kernel": str(object_path),
        "object_path": str(object_path),
        "source": str(out / "generated.asc"),
        "compiled_source": str(source_path),
        "prim_func": str(out / "segment.ir.json"),
        "output_dir": str(out),
        "abi": {
            "inputs": [{"name": "A", "shape": [64, 64], "dtype": "bfloat16"}, {"name": "B", "shape": [64, 64], "dtype": "bfloat16"}],
            "output": {"name": "C", "shape": [64, 64], "dtype": "float32"},
            "parameter_offsets": [0, 8, 16],
            "parameter_bytes": 24,
            "helper_argument_bytes": 32,
            "blocks": 1,
            "tiling_key": 0,
            "operation": "A @ B",
        },
        "commands": commands,
        "sha256": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ("segment.tir", "segment.ir.json", "generated.asc", "compiled.asc", "segment.aibin")
        },
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    return metadata
