"""Source-TIR checks for distributed ownership and structured event lifetimes.

This validates established local violations. Peer matching and progress remain
protocol obligations, checked by the optional co-simulation transport.
"""

import json
import math

from tvm import IRModule, arith, ir, tirx


PREFIX = "[TileLang Distributed]"


def fail(message, node=None):
    span = getattr(node, "span", None)
    location = f" ({span.source_name.name}:{span.line})" if span is not None else ""
    raise ValueError(f"{PREFIX} {message}{location}")


def op_name(call):
    return call.op.name if isinstance(call, tirx.Call) and isinstance(call.op, ir.Op) else ""


def is_distributed(call):
    return op_name(call).startswith("tl.distributed.")


def text(value):
    return value.value if isinstance(value, tirx.StringImm) else str(value)


def manifest(func):
    value = func.attrs.get("tl.distributed.manifest") if func.attrs else None
    return json.loads(str(value)) if value is not None else None


def reject_native_distributed(func_or_mod):
    """Gate an unavailable execution capability independently of semantic flags.

    Check both metadata and operations: removing the manifest from source IR
    must not send communication intrinsics into an ordinary native codegen.
    This does not validate or reject the explicit local co-simulation path.
    """
    functions = func_or_mod.functions.values() if isinstance(func_or_mod, IRModule) else (func_or_mod,)
    for func in functions:
        if not isinstance(func, tirx.PrimFunc):
            continue
        found = bool(func.attrs and "tl.distributed.manifest" in func.attrs)
        if not found:

            def visit(node):
                nonlocal found
                found |= is_distributed(node)

            tirx.stmt_functor.post_order_visit(func.body, visit)
        if found:
            fail(
                "native transport is not certified for distributed source IR; "
                "use tilelang.distributed.compile(..., profile='ascend950_cannsim_v1') "
                "for explicit local protocol and CANN computation co-simulation"
            )


def tile_region(value):
    """Decode ordinary TileOp regions without interpreting their data."""
    if isinstance(value, tirx.BufferRegion):
        return value
    if isinstance(value, tirx.Buffer):
        return tirx.BufferRegion(value, [ir.Range.from_min_extent(0, extent) for extent in value.shape])
    if op_name(value) == "tl.region":
        load = value.args[0]
        if isinstance(load, tirx.BufferLoad):
            return tirx.BufferRegion(
                load.buffer, [ir.Range.from_min_extent(start, extent) for start, extent in zip(load.indices, value.args[2:])]
            )
    return None


def statements(node):
    if isinstance(node, tirx.SeqStmt):
        return list(node.seq)
    if isinstance(node, tirx.SBlockRealize):
        return [node.block]
    if isinstance(node, tirx.IfThenElse):
        return [node.then_case] + ([node.else_case] if node.else_case is not None else [])
    if isinstance(node, (tirx.For, tirx.While, tirx.AttrStmt, tirx.SBlock)):
        return [node.body]
    return []


def check_distributed(func):
    spec = manifest(func)
    calls = []
    tirx.stmt_functor.post_order_visit(func.body, lambda n: calls.append(n) if is_distributed(n) else None)
    if not calls and spec is None:
        return
    if spec is None:
        fail("communication IR lacks its canonical ABI manifest; construct it with the eager frontend")
    specs = {}
    for entry in spec.get("objects", []):
        index = entry.get("param_index", -1)
        if not isinstance(index, int) or not 0 <= index < len(func.params):
            fail("ABI manifest has an invalid symmetric object parameter index")
        buffer = func.buffer_map.get(func.params[index])
        if buffer is None or buffer.scope() != "global":
            fail("symmetric objects require ordinary global Buffer parameters")
        if str(buffer.dtype) != entry["dtype"] or len(buffer.shape) != len(entry["shape"]):
            fail(f"ABI manifest does not match Buffer {buffer.name}")
        if any(not isinstance(dim, tirx.IntImm) or int(dim) != expected for dim, expected in zip(buffer.shape, entry["shape"])):
            fail(f"ABI shape does not match Buffer {buffer.name}")
        strides, running = [], 1
        for dim in reversed(entry["shape"]):
            strides.append(running)
            running *= dim
        strides = list(reversed(strides))
        actual_strides = list(map(int, buffer.strides)) if len(buffer.strides) else strides
        if actual_strides != entry["strides"]:
            fail(f"ABI physical strides do not match Buffer {buffer.name}")
        if entry["kind"] == "signal":
            if buffer.dtype != "int32" or actual_strides != [value * 16 for value in strides]:
                fail("P0 SignalBuffer requires int32 cells with independent 64-byte physical strides")
            if entry["physical_nbytes"] != math.prod(entry["shape"]) * 64:
                fail("SignalBuffer ABI storage must include all isolated 64-byte cells")
        elif entry["kind"] != "symmetric":
            fail("unknown symmetric object kind in ABI manifest")
        if entry["remote_access"] not in ("read", "write", "read_write"):
            fail("unknown remote permission in ABI manifest")
        if buffer.data in specs:
            fail("ABI manifest repeats a symmetric object parameter")
        specs[buffer.data] = entry
    groups = {entry["name"]: entry["size"] for entry in spec.get("groups", [])}
    channels = {entry["name"]: entry["group"] for entry in spec.get("channels", [])}
    signal_data = {data for data, entry in specs.items() if entry["kind"] == "signal"}
    analyzer = arith.Analyzer()

    def validate_i32(value, what, node, *, positive=False):
        if not isinstance(value, tirx.PrimExpr) or value.dtype != "int32":
            fail(f"{what} must be int32 without implicit narrowing", node)
        if positive and analyzer.can_prove(value <= 0):
            fail("published signal value must be in 1..2^31-1", node)

    def validate_peer(peer, group, node):
        validate_i32(peer, "peer", node)
        if analyzer.can_prove(peer < 0) or analyzer.can_prove(peer >= groups[group]):
            fail("remote peer is outside the declared group", node)
        reads = []
        tirx.stmt_functor.post_order_visit(peer, lambda n: reads.append(n) if isinstance(n, tirx.BufferLoad) else None)
        if reads:
            fail("P0 does not support data-dependent peers; derive peer from rank, size and compile-time constants", node)

    def validate_region(region, group, node, *, signal=False, remote=False, access="read"):
        if not isinstance(region, tirx.BufferRegion):
            fail("communication effects require actual BufferRegion annotations", node)
        entry = specs.get(region.buffer.data)
        if entry is None:
            fail(f"communication Buffer {region.buffer.name} is missing its symmetric declaration", node)
        if entry["group"] != group:
            fail("communication region and channel use different groups", node)
        if (entry["kind"] == "signal") != signal:
            fail("SignalBuffer may only be used by signal APIs, and signal APIs require SignalBuffer", node)
        if remote and entry["remote_access"] not in (access, "read_write"):
            fail(f"remote {access} violates {entry['remote_access']} permission for {region.buffer.name}", node)
        if len(region.region) != len(region.buffer.shape):
            fail("communication region must specify every buffer dimension", node)
        spans_row = False
        for extent, bound in zip(region.region, region.buffer.shape):
            if analyzer.can_prove(extent.extent <= 0):
                fail("communication regions must be nonempty", node)
            if analyzer.can_prove(extent.min < 0) or analyzer.can_prove(extent.min + extent.extent > bound):
                fail(f"communication region for {region.buffer.name} is out of bounds", node)
            if signal:
                if not analyzer.can_prove(extent.extent == 1):
                    fail("signal operation must select exactly one signal cell", node)
            else:
                if spans_row and not (analyzer.can_prove(extent.min == 0) and analyzer.can_prove(extent.extent == bound)):
                    fail("P0 profile only supports contiguous regions; explicitly pack this tile", node)
                spans_row |= not analyzer.can_prove(extent.extent == 1)

    def validate_call(call):
        name, anno = op_name(call).split(".")[-1], call.annotations
        arity = {"rank": 1, "copy_async": 3, "put_signal": 3, "wait": 2, "signal": 2, "wait_signal": 1, "fence": 0, "quiet": 0}
        if name not in arity or len(call.args) != arity[name]:
            fail(f"unsupported distributed operation or invalid argument count: {name}", call)
        if name == "rank":
            if len(call.args) != 1 or text(call.args[0]) not in groups:
                fail("rank requires a group declared in the ABI manifest", call)
            return
        required = ("schema_version", "group", "group_size", "channel")
        if any(key not in anno for key in required) or int(anno["schema_version"]) != 1:
            fail("communication call is missing its P0 schema annotations", call)
        group, channel = text(anno["group"]), text(anno["channel"])
        if group not in groups or int(anno["group_size"]) != groups[group] or channels.get(channel) != group:
            fail("communication group/channel does not match the ABI manifest", call)
        if name in ("copy_async", "put_signal"):
            if not isinstance(call.args[0], tirx.IntImm) or int(call.args[0]) < 0:
                fail("communication event identity must be a nonnegative static integer", call)
            direction = text(anno.get("direction", ""))
            if name == "copy_async":
                src_local = isinstance(call.args[1], tirx.IntImm) and int(call.args[1]) == -1
                dst_local = isinstance(call.args[2], tirx.IntImm) and int(call.args[2]) == -1
                if src_local == dst_local:
                    fail("copy_async requires exactly one remote endpoint", call)
                if direction != ("put" if src_local else "get"):
                    fail("copy direction disagrees with endpoint ownership", call)
                validate_peer(call.args[2] if src_local else call.args[1], group, call)
            else:
                if direction != "put":
                    fail("put_signal must have put direction", call)
                validate_peer(call.args[1], group, call)
                validate_i32(call.args[2], "signal value", call, positive=True)
                validate_region(anno.get("signal_region"), group, call, signal=True, remote=True, access="write")
            validate_region(anno.get("src_region"), group, call, remote=direction == "get", access="read")
            validate_region(anno.get("dst_region"), group, call, remote=direction == "put", access="write")
        elif name in ("signal", "wait_signal"):
            validate_region(anno.get("signal_region"), group, call, signal=True, remote=name == "signal", access="write")
            validate_i32(call.args[-1], "signal value", call, positive=True)
            if name == "signal":
                validate_peer(call.args[0], group, call)
                if text(anno.get("order", "")) != "release":
                    fail("signal only supports release ordering", call)
                for region in anno.get("release_regions", []):
                    validate_region(region, group, call)
            else:
                if text(anno.get("order", "")) != "acquire" or text(anno.get("cmp", "")) != "ge":
                    fail("wait_signal only supports acquire ordering and ge comparison", call)
                covers, peers = anno.get("covers", []), anno.get("covers_peers", [])
                if len(covers) != len(peers):
                    fail("wait_signal covers and covers_peers must have equal lengths", call)
                for region, peer in zip(covers, peers):
                    local = isinstance(peer, tirx.IntImm) and int(peer) == -1
                    if not local:
                        validate_peer(peer, group, call)
                    validate_region(region, group, call, remote=not local)

    for call in calls:
        validate_call(call)

    def overlap(a, b):
        if not a.buffer.data.same_as(b.buffer.data):
            return False
        if len(a.region) != len(b.region):
            return None
        if any(
            analyzer.can_prove(x.min + x.extent <= y.min) or analyzer.can_prove(y.min + y.extent <= x.min)
            for x, y in zip(a.region, b.region)
        ):
            return False
        if all(
            analyzer.can_prove(x.min < y.min + y.extent) and analyzer.can_prove(y.min < x.min + x.extent)
            for x, y in zip(a.region, b.region)
        ):
            return True
        return None

    def check_access(region, writing, events, node):
        if region.buffer.data in signal_data:
            fail(f"SignalBuffer {region.buffer.name} only accepts TD.signal / TD.wait_signal; ordinary data access is forbidden", node)
        for entry in events.values():
            if entry["done"]:
                continue
            protected = entry["src"] if entry["direction"] == "put" else entry["dst"]
            if (writing or entry["direction"] == "get") and overlap(region, protected) is True:
                fail(
                    f"{entry['direction']} event {entry['id']} still owns {region.buffer.name}; TD.wait(..., completion='local') before this access",
                    node,
                )

    def check_regular(node, events):
        def visit(value):
            if is_distributed(value) and op_name(value) != "tl.distributed.rank":
                fail("communication operations require standalone protocol statements", value)
            if isinstance(value, tirx.BufferStore):
                check_access(
                    tirx.BufferRegion(value.buffer, [ir.Range.from_min_extent(index, 1) for index in value.indices]), True, events, value
                )
            elif isinstance(value, tirx.BufferLoad):
                check_access(
                    tirx.BufferRegion(value.buffer, [ir.Range.from_min_extent(index, 1) for index in value.indices]), False, events, value
                )
            elif isinstance(value, tirx.Call) and not is_distributed(value):
                for index, arg in enumerate(value.args):
                    region = tile_region(arg)
                    if region is not None:
                        if op_name(arg) == "tl.region":
                            writing = int(arg.args[1]) & 2
                        else:
                            writing = op_name(value) == "tl.tileop.copy" and index == 1
                        check_access(region, bool(writing), events, value)
                for arg in value.args:
                    if isinstance(arg, tirx.Var) and arg in signal_data:
                        fail("SignalBuffer pointer escape is forbidden", value)

        tirx.stmt_functor.post_order_visit(node, visit)

    def walk(node, events, path):
        if isinstance(node, tirx.SeqStmt):
            for child in node.seq:
                walk(child, events, path)
        elif isinstance(node, tirx.Evaluate) and is_distributed(node.value):
            call, name = node.value, op_name(node.value).split(".")[-1]
            anno = call.annotations
            if name in ("copy_async", "put_signal"):
                event_id = int(call.args[0])
                if event_id in events:
                    fail(f"duplicate communication event {event_id}", call)
                direction = text(anno.get("direction", "put"))
                if direction not in ("put", "get"):
                    fail("copy direction must be put or get", call)
                src, dst = anno["src_region"], anno["dst_region"]
                if src.buffer.dtype != dst.buffer.dtype:
                    fail("copy dtype mismatch; explicit conversion must be local", call)
                count_src, count_dst = 1, 1
                for item in src.region:
                    count_src *= item.extent
                for item in dst.region:
                    count_dst *= item.extent
                if analyzer.can_prove(count_src != count_dst):
                    fail("copy element counts differ", call)
                if direction == "put":
                    check_access(src, False, events, call)
                else:
                    check_access(dst, True, events, call)
                events[event_id] = dict(
                    id=event_id, direction=direction, src=src, dst=dst, channel=text(anno["channel"]), path=path, done=False
                )
            elif name == "wait":
                event_id, completion = int(call.args[0]), text(call.args[1])
                if event_id not in events or events[event_id]["path"] != path:
                    fail(f"event {event_id} escapes its structured branch or loop iteration; issue and wait in the same scope", call)
                event = events[event_id]
                if event["channel"] != text(anno["channel"]):
                    fail("event wait uses another channel", call)
                if completion not in ("local", "remote") or (completion == "remote" and event["direction"] == "get"):
                    fail("get does not support completion='remote'; use local completion", call)
                event["done"] = True
            elif name in ("signal", "quiet"):
                for event in events.values():
                    if event["channel"] == text(anno["channel"]):
                        event["done"] = True
            elif name not in ("wait_signal", "fence"):
                fail(f"unsupported distributed statement {name}", call)
        elif isinstance(node, (tirx.For, tirx.IfThenElse, tirx.While)):
            if isinstance(node, tirx.For):
                check_regular(tirx.Evaluate(node.min), events)
                check_regular(tirx.Evaluate(node.extent), events)
            else:
                check_regular(tirx.Evaluate(node.condition), events)
            for index, child in enumerate(statements(node)):
                nested = {key: dict(value) for key, value in events.items()}
                child_path = (*path, id(node), index)
                walk(child, nested, child_path)
                for event_id in nested.keys() - events.keys():
                    if not nested[event_id]["done"]:
                        fail(f"event {event_id} escapes a structured branch or loop iteration; wait or quiet inside its scope", node)
        elif isinstance(node, tirx.Bind):
            if not is_distributed(node.value):
                check_regular(tirx.Evaluate(node.value), events)
            elif op_name(node.value) != "tl.distributed.rank":
                fail("communication statements cannot be used as ordinary scalar expressions", node)
        elif isinstance(node, (tirx.SBlockRealize, tirx.SBlock, tirx.AttrStmt)):
            for child in statements(node):
                walk(child, events, path)
        else:
            check_regular(node, events)

    walk(func.body, {}, ())


def DistributedChecker():
    def check(func, mod, ctx):
        check_distributed(func)
        return func

    return tirx.transform.prim_func_pass(check, opt_level=0)
