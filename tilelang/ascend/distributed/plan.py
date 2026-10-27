"""Plan source TD operations and local computation without loading a transport.

Both the local protocol model and the native phased profile consume this same
validated plan. Scalar control flow is resolved on the host; device data never
selects a host branch. Communication completion remains a transport obligation.
"""

from dataclasses import dataclass
import math

import numpy as np
import tvm
from tvm import tirx

from tilelang.analysis.distributed_checker import check_distributed, fail, is_distributed, op_name, text, tile_region


def _integer(value, dtype):
    dtype = tvm.DataType(dtype)
    if not isinstance(value, (int, np.integer, bool)) or dtype.lanes != 1 or dtype.type_code not in (0, 1, 6):
        fail("local phase control requires scalar integer values")
    bits = dtype.bits
    lower, upper = (0, 1 << bits) if dtype.type_code != 0 else (-(1 << (bits - 1)), 1 << (bits - 1))
    if dtype.type_code == 6:
        lower, upper = 0, 2
    if not lower <= value < upper:
        fail(f"integer overflow or narrowing to {dtype} is unsupported in local phase control")
    return int(value)


def _trunc_div(a, b):
    quotient = abs(a) // abs(b)
    return -quotient if (a < 0) != (b < 0) else quotient


def _eval(expr, env, rank):
    if isinstance(expr, (int, bool)):
        return int(expr)
    if isinstance(expr, tirx.IntImm):
        return expr.value
    if isinstance(expr, tirx.Var):
        if expr not in env:
            fail(f"unbound scalar {expr.name}; supply it in launch scalars")
        return _integer(env[expr], expr.dtype)
    if isinstance(expr, tirx.Cast):
        return _integer(_eval(expr.value, env, rank), expr.dtype)
    if isinstance(expr, tirx.Select):
        return _eval(expr.true_value if _eval(expr.condition, env, rank) else expr.false_value, env, rank)
    if isinstance(expr, tirx.Not):
        return not _eval(expr.a, env, rank)
    if isinstance(expr, tirx.Call):
        if op_name(expr) == "tl.distributed.rank":
            return rank
        if op_name(expr) == "tl.launch_thread_idx":
            return 0
        fail(f"data-dependent scalar or unsupported call {op_name(expr)} in local profile", expr)
    operations = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "FloorDiv": lambda a, b: a // b,
        "FloorMod": lambda a, b: a % b,
        "Div": _trunc_div,
        "Mod": lambda a, b: a - _trunc_div(a, b) * b,
        "LT": lambda a, b: a < b,
        "LE": lambda a, b: a <= b,
        "GT": lambda a, b: a > b,
        "GE": lambda a, b: a >= b,
        "EQ": lambda a, b: a == b,
        "NE": lambda a, b: a != b,
        "And": lambda a, b: bool(a and b),
        "Or": lambda a, b: bool(a or b),
        "Min": min,
        "Max": max,
    }
    name = type(expr).__name__
    if name in operations:
        try:
            result = operations[name](_eval(expr.a, env, rank), _eval(expr.b, env, rank))
        except ZeroDivisionError:
            fail("division by zero in local phase control", expr)
        return _integer(result, expr.dtype)
    fail(f"unsupported scalar expression {type(expr).__name__} in local profile", expr)


@dataclass(frozen=True)
class Region:
    rank: int
    key: str
    offset: int
    count: int
    shape: tuple

    @property
    def identity(self):
        return self.rank, self.key, self.offset, self.count


@dataclass(frozen=True)
class SignalRef:
    rank: int
    key: str
    index: int


def plan_program(program, *, rank, scalars, world_size, allow_vector=False):
    check_distributed(program)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        fail("rank must be inside the static group")
    params = [param for param in program.params if param not in program.buffer_map]
    if set(scalars) - {param.name for param in params} - {"epoch0"}:
        fail("launch includes unknown scalar parameters")
    if any(param.name not in scalars for param in params):
        fail("launch is missing a scalar parameter")
    env = {param: _integer(scalars[param.name], param.dtype) for param in params}
    plan, allocations, local, block_annotations = [], [], [], {}
    regions = {}

    def region(source, owner=rank):
        shape = tuple(int(_eval(x.extent, env, rank)) for x in source.region)
        starts = tuple(int(_eval(x.min, env, rank)) for x in source.region)
        full = tuple(int(x) for x in source.buffer.shape)
        if any(n <= 0 or start < 0 or start + n > size for start, n, size in zip(starts, shape, full)):
            fail(f"out-of-bounds or empty region {source.buffer.name}: start={starts}, shape={shape}")
        stride, offset, count = 1, 0, math.prod(shape)
        for start, size in zip(reversed(starts), reversed(full)):
            offset += start * stride
            stride *= size
        # In row-major storage, the first and last selected elements must
        # span exactly count entries. Signals are separate logical cells.
        last, stride = 0, 1
        for start, n, size in zip(reversed(starts), reversed(shape), reversed(full)):
            last += (start + n - 1) * stride
            stride *= size
        if last - offset + 1 != count:
            fail(f"non-contiguous region {source.buffer.name} is unsupported; explicitly pack a tile")
        if owner < 0 or owner >= world_size:
            fail(f"peer {owner} is outside the group")
        result = Region(owner, source.buffer.name, offset, count, shape)
        regions[result.identity] = result
        return result

    def signal_ref(source, owner=rank):
        result = region(source, owner)
        if result.count != 1:
            fail("signal API requires one isolated signal cell")
        return SignalRef(owner, result.key, result.offset)

    def flush():
        if local:
            fail("local computation must end in a complete GM output copy before a communication boundary")

    def local_op(stmt):
        call = stmt.value
        if op_name(call) not in ("tl.tileop.ascend_copy", "tl.tileop.copy", "tl.tileop.gemm"):
            fail(f"unsupported local operation {op_name(call)}; use copy/GEMM phases in this profile", call)
        local.append(stmt)
        if "copy" not in op_name(call):
            return
        src, dst = tile_region(call.args[0]), tile_region(call.args[1])
        if src is None or dst is None:
            fail("copy lacks explicit regions")
        if dst.buffer.scope() != "global":
            return
        output = region(dst)
        if len(local) == 1 and src.buffer.scope() == "global":
            source = region(src)
            if call.annotations or src.buffer.dtype != dst.buffer.dtype or source.count != output.count:
                fail("local GM copy requires equal dtype/count and default copy semantics; pack or convert in an explicit supported phase")
            plan.append(("local_copy", source, output))
        else:
            inputs = []
            for item in local:
                if isinstance(item, tirx.Evaluate) and "copy" in op_name(item.value):
                    candidate = tile_region(item.value.args[0])
                    if candidate.buffer.scope() == "global":
                        inputs.append(candidate)
            has_gemm = any(isinstance(item, tirx.Evaluate) and op_name(item.value) == "tl.tileop.gemm" for item in local)
            if not has_gemm and not allow_vector:
                fail("local GEMM phase requires exactly two explicit GM inputs and one GM output")
            if has_gemm and len(inputs) != 2:
                fail("local GEMM phase requires exactly two explicit GM inputs and one GM output")
            # Substitute launch/loop scalars before extraction. The segment
            # compiler consumes these exact operations, not a canned GEMM.
            replacements = {key: tirx.const(value, key.dtype) for key, value in env.items()}
            body = [tirx.stmt_functor.substitute(item, replacements) for item in local]

            def fixed(value):
                return tirx.BufferRegion(
                    value.buffer,
                    [tvm.ir.Range.from_min_extent(_eval(r.min, env, rank), _eval(r.extent, env, rank)) for r in value.region],
                )

            from .segments import extract_segment

            # Validate every local phase before the first protocol event,
            # including in protocol-only mode.
            if has_gemm:
                extract_segment(body, list(allocations), [fixed(x) for x in inputs], fixed(dst), annotations=dict(block_annotations))
            else:
                from .fused import validate_vector_phase

                validate_vector_phase(body, list(allocations), [fixed(x) for x in inputs], fixed(dst))
            plan.append(
                (
                    "gemm" if has_gemm else "vector",
                    tuple(region(x) for x in inputs),
                    output,
                    body,
                    list(allocations),
                    [fixed(x) for x in inputs],
                    fixed(dst),
                    dict(block_annotations),
                )
            )
        local.clear()

    def walk(node):
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                walk(child)
        elif isinstance(node, tirx.SBlockRealize):
            if node.iter_values or not _eval(node.predicate, env, rank):
                fail("local phases require unconditional root/allocation blocks without block iteration bindings")
            walk(node.block)
        elif isinstance(node, tirx.SBlock):
            if allow_vector and node.name_hint == "SIMD_VF":
                replacements = {key: tirx.const(value, key.dtype) for key, value in env.items()}
                local.append(tirx.stmt_functor.substitute(node, replacements))
                return
            if node.iter_vars or node.match_buffers:
                fail("block iteration variables and match_buffer aliases are unsupported in local phases")
            unknown = set(node.annotations) - {"layout_map"}
            if unknown:
                fail(f"unsupported local phase block annotations: {sorted(unknown)}")
            allocations.extend(node.alloc_buffers)
            block_annotations.update(dict(node.annotations))
            if node.init is not None:
                fail("SBlock init is unsupported in local distributed phases")
            walk(node.body)
        elif isinstance(node, tirx.Bind):
            env[node.var] = _eval(node.value, env, rank)
        elif isinstance(node, tirx.For):
            flush()
            if "td.gemm" in node.annotations:
                from .tiled_gemm import parse_tiled_gemm

                replacements = {key: tirx.const(value, key.dtype) for key, value in env.items()}
                fixed_loop = tirx.stmt_functor.substitute(node, replacements)
                body, inputs, output = parse_tiled_gemm(fixed_loop, list(allocations), dict(block_annotations))
                plan.append(
                    (
                        "gemm",
                        tuple(region(x) for x in inputs),
                        region(output),
                        [body],
                        list(allocations),
                        inputs,
                        output,
                        dict(block_annotations),
                    )
                )
                return
            minimum, extent = int(_eval(node.min, env, rank)), int(_eval(node.extent, env, rank))
            if node.thread_binding is not None:
                if node.thread_binding.thread_tag != "blockIdx.x" or extent != 1 or minimum != 0:
                    fail("local distributed profile requires T.Kernel(1), with one logical issuer")
            elif node.kind != tirx.ForKind.SERIAL:
                fail("local distributed profile supports serial loops; scheduled/pipelined communication is unavailable")
            if node.annotations:
                fail("annotated loops have no validated local phase semantics; scheduled/pipelined communication is unavailable")
            if extent < 0 or extent > 10000:
                fail("local profile loop extent must be between 0 and 10000")
            for index in range(minimum, minimum + extent):
                env[node.loop_var] = index
                walk(node.body)
            flush()
        elif isinstance(node, tirx.IfThenElse):
            flush()
            selected = node.then_case if _eval(node.condition, env, rank) else node.else_case
            if selected is not None:
                walk(selected)
            flush()
        elif isinstance(node, tirx.AttrStmt):
            fail(f"attribute {node.attr_key} has no validated local phase semantics; scheduled communication is unavailable")
        elif isinstance(node, tirx.Evaluate):
            call = node.value
            if isinstance(call, tirx.IntImm):
                return
            if not is_distributed(call):
                local_op(node)
                return
            flush()
            name, anno = op_name(call).split(".")[-1], call.annotations
            args = [_eval(x, env, rank) if not isinstance(x, tirx.StringImm) else text(x) for x in call.args]
            if name in ("copy_async", "put_signal"):
                source_peer, target_peer = (args[1], args[2]) if name == "copy_async" else (-1, args[1])
                if (source_peer == -1) == (target_peer == -1):
                    fail("copy requires exactly one remote endpoint")
                peer = target_peer if source_peer == -1 else source_peer
                if peer == rank:
                    fail("remote peer is the local rank; use an ordinary local copy")
                source = region(anno["src_region"], rank if source_peer == -1 else source_peer)
                destination = region(anno["dst_region"], rank if target_peer == -1 else target_peer)
                if name == "put_signal":
                    plan.append((name, args[0], source, destination, signal_ref(anno["signal_region"], peer), args[2]))
                else:
                    plan.append((name, args[0], source, destination))
            elif name == "wait":
                plan.append((name, args[0], args[1]))
            elif name == "signal":
                if args[0] == rank:
                    fail("signal must target a remote rank")
                plan.append(
                    (
                        name,
                        signal_ref(anno["signal_region"], args[0]),
                        args[1],
                        tuple(region(x) for x in anno.get("release_regions", [])),
                    )
                )
            elif name == "wait_signal":
                covers = tuple(
                    region(value, rank if _eval(peer, env, rank) == -1 else _eval(peer, env, rank))
                    for value, peer in zip(anno["covers"], anno["covers_peers"])
                )
                plan.append((name, signal_ref(anno["signal_region"]), args[0], covers))
            elif name in ("fence", "quiet"):
                plan.append((name,))
            else:
                fail(f"unsupported statement {name}")
        else:
            fail(f"unsupported statement {type(node).__name__} in local distributed profile", node)

    walk(program.body)
    flush()
    plan.append(("quiet",))  # Mandatory normal-exit communication drain.
    return plan, regions
