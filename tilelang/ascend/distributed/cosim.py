"""Lower canonical distributed TIR to explicit local protocol/compute phases.

This is an opt-in functional adapter, not a native multi-device execution
backend. The ordinary Ascend pipeline still compiles each extracted local TIR.
"""

from dataclasses import dataclass
from contextlib import suppress
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile

import numpy as np
import tvm
from tvm import tirx

from tilelang.analysis.distributed_checker import check_distributed, fail, is_distributed, manifest, op_name, text, tile_region
from .capabilities import COSIM_PROFILE, require_cosim_profile


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _process_identity(pid):
    """Read Linux process ancestry, group and birth time without trusting PID reuse."""
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1]), int(fields[2]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _worker_descendants(pid):
    processes = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            identity = _process_identity(entry.name)
            if identity is not None:
                processes[int(entry.name)] = identity
    owned = {pid}
    while True:
        children = {child for child, (parent, _, _) in processes.items() if parent in owned}
        if children <= owned:
            break
        owned.update(children)
    return {(child, processes[child][1], processes[child][2]) for child in owned if child in processes}


def _signal_worker_groups(owned, signum):
    groups = set()
    for pid, group, birth in owned:
        current = _process_identity(pid)
        if current is not None and current[1:] == (group, birth) and group > 1:
            groups.add(group)
    for group in groups:
        with suppress(ProcessLookupError):
            os.killpg(group, signum)


class _Worker:
    def __init__(self, helper, kernel, directory, cores, timeout):
        self.directory, self.timeout = directory, timeout
        self.log = (directory / "worker.log").open("w")
        self.process = subprocess.Popen(
            [
                str(helper / ".venv/bin/python"),
                "-m",
                "cosim.worker",
                "--kernel",
                str(kernel),
                "--run-dir",
                str(directory),
                "--timeout",
                str(timeout),
                "--expected-cores",
                *cores,
            ],
            cwd=helper,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            response = self.read()
        except Exception:
            self.close()
            raise
        if not response.get("ready"):
            self.close()
            raise RuntimeError(response.get("error", "CANN worker initialization failed"))

    def read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout + 30):
                self.close()
                raise RuntimeError("CANN worker exceeded bounded response time; see worker.log")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("CANN worker exited without a response; see worker.log")
        return json.loads(line)

    def execute(self, rank, index, a, b):
        request_dir = self.directory / f"request_{index}"
        request_dir.mkdir()
        a.tofile(request_dir / "a.bin")
        b.tofile(request_dir / "b.bin")
        self.process.stdin.write(
            json.dumps(
                dict(op="execute", rank=rank, segment_id=f"segment{index:04d}", a=str(request_dir / "a.bin"), b=str(request_dir / "b.bin"))
            )
            + "\n"
        )
        self.process.stdin.flush()
        return self.read()

    def close(self):
        owned = _worker_descendants(self.process.pid) if self.process.poll() is None else set()
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                owned.update(_worker_descendants(self.process.pid))
                try:
                    # The worker's SIGTERM handler unwinds the executor, which
                    # records interruption and kills the native runner group.
                    # Include build children that share the worker's group.
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    owned.update(_worker_descendants(self.process.pid))
                    _signal_worker_groups(owned, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
        # A native runner uses its own session. Keep identities captured before
        # the worker exits so a stuck worker cannot orphan that runner on SIGKILL.
        _signal_worker_groups(owned, signal.SIGKILL)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self.log.close()


class CompiledProgram:
    def __init__(
        self,
        program,
        *,
        profile=COSIM_PROFILE,
        target="ascend",
        mode="rank-segments",
        helper_root=None,
        output_dir=None,
        compile_profile="cann-9.1-gemm64",
    ):
        require_cosim_profile(profile)
        if target != "ascend":
            fail("local distributed profile supports the Ascend 950 AscendC target only")
        if mode not in ("protocol", "rank-segments"):
            fail("mode must be protocol or rank-segments")
        check_distributed(program)
        self.prim_func = program
        self.manifest = manifest(program)
        if self.manifest is None:
            fail("expected an eager distributed PrimFunc")
        if len(self.manifest["groups"]) != 1 or len(self.manifest["channels"]) != 1:
            fail("local P0 profile currently supports one static group and one channel")
        self.ir_json = tvm.ir.save_json(program)
        self.manifest = {
            **self.manifest,
            "program_sha256": hashlib.sha256(self.ir_json.encode()).hexdigest(),
            "local_compile_profile": compile_profile,
        }
        self.mode, self.compile_profile = mode, compile_profile
        self.output_dir = Path(output_dir or "output/distributed").expanduser().resolve()
        configured = helper_root or os.environ.get("CANNSIM_ROOT")
        if not configured:
            fail("set CANNSIM_ROOT or pass helper_root pointing to the installed cannsim-helper repository")
        self.helper = Path(configured).expanduser().resolve()
        if not (self.helper / "cosim/protocol.py").is_file():
            fail(f"missing extended cannsim-helper at {self.helper}")
        sys.path.insert(0, str(self.helper))
        try:
            self.model = importlib.import_module("cosim.protocol")
        finally:
            sys.path.remove(str(self.helper))
        if Path(self.model.__file__).resolve() != self.helper / "cosim/protocol.py":
            fail("another cosim module is already imported; use a fresh process")

    def _plan(self, rank, scalars, arrays):
        params = [param for param in self.prim_func.params if param not in self.prim_func.buffer_map]
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
            if owner < 0 or owner >= len(arrays):
                fail(f"peer {owner} is outside the group")
            result = Region(owner, source.buffer.name, offset, count, shape)
            regions[result.identity] = result
            return result

        def signal_ref(source, owner=rank):
            result = region(source, owner)
            if result.count != 1:
                fail("signal API requires one isolated signal cell")
            return self.model.SignalRef(owner, result.key, result.offset)

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
                    fail(
                        "local GM copy requires equal dtype/count and default copy semantics; pack or convert in an explicit supported phase"
                    )
                plan.append(("local_copy", source, output))
            else:
                inputs = []
                for item in local:
                    if "copy" in op_name(item.value):
                        candidate = tile_region(item.value.args[0])
                        if candidate.buffer.scope() == "global":
                            inputs.append(candidate)
                if len(inputs) != 2 or not any(op_name(item.value) == "tl.tileop.gemm" for item in local):
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
                extract_segment(body, list(allocations), [fixed(x) for x in inputs], fixed(dst), annotations=dict(block_annotations))
                plan.append(
                    (
                        "gemm",
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

        walk(self.prim_func.body)
        flush()
        plan.append(("quiet",))  # Mandatory normal-exit communication drain.
        return plan, regions

    def run(
        self,
        *,
        session,
        ticket,
        buffers=None,
        scalars=None,
        seed=0,
        fault=None,
        slow_rank=1,
        consumer_delay=7,
        segment_timeout=120,
        **kwargs,
    ):
        if kwargs:
            fail(f"unknown local launch options: {sorted(kwargs)}")
        scalar_values = dict(scalars or {})
        scalar_values.setdefault("epoch0", ticket.base)
        arrays = {rank: {key: obj.arrays[rank] for key, obj in session.objects.items()} for rank in range(session.world_size)}
        for key, value in (buffers or {}).items():
            if hasattr(value, "arrays"):
                value = value.arrays
            for rank in arrays:
                arrays[rank][key] = value[rank]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"p0_{self.mode.replace('-', '_')}_", dir=self.output_dir))
        print(f"Distributed run: {directory}", flush=True)
        (directory / "program.json").write_text(self.ir_json)
        (directory / "program.tir").write_text(self.prim_func.script(show_meta=True))
        _json(directory / "abi.json", self.manifest)
        model = self.model
        simulator = model.Simulator(seed=seed, fault=fault)
        report = dict(
            passed=False,
            mode=self.mode,
            profile=COSIM_PROFILE,
            run_dir=str(directory),
            program_sha256=self.manifest["program_sha256"],
            communication="CPU model from canonical distributed TIR",
            kernel_executed=False,
            segments=[],
            compiled_segments=[],
            errors=[],
            seed=seed,
            fault=fault,
            launch_id=ticket.launch_id,
            epoch_base=ticket.base,
            epoch_end=ticket.end,
            time_domains="protocol logical ticks; independent per-segment CANN Core 0 traces",
        )
        workers, compiled, all_regions = {}, {}, {}

        def ref(region):
            return model.BufferRef(region.rank, f"{region.key}@{region.offset}+{region.count}")

        try:
            plans = {}
            for rank in arrays:
                plans[rank], regions = self._plan(rank, scalar_values, arrays)
                all_regions.update(regions)
                # Validate the whole rank program before registering/advancing
                # any event, so inconsistent launches cannot partly execute.
                for instruction in plans[rank]:
                    kind, *args = instruction
                    peers, published, observed = [], None, None
                    if kind in ("copy_async", "put_signal"):
                        peers = [r.rank for r in args[1:3] if r.rank != rank]
                        if kind == "put_signal":
                            published = args[4]
                    elif kind == "signal":
                        peers, published = [args[0].rank], args[1]
                    elif kind == "wait_signal":
                        peers, observed = [r.rank for r in args[2] if r.rank != rank], args[1]
                    if any(peer not in ticket.channel.peers[rank] for peer in peers):
                        fail(f"rank {rank} requests a peer outside prepared channel {ticket.channel.key}")
                    if published is not None and not ticket.base < published <= ticket.end:
                        fail(f"published epoch {published} is outside the prepared interval ({ticket.base}, {ticket.end}]")
                    if observed is not None and not 1 <= observed <= ticket.end:
                        fail(f"wait epoch {observed} is outside the positive prepared epoch range")
            signal_keys = {entry["key"] for entry in self.manifest["objects"] if entry["kind"] == "signal"}
            object_specs = {entry["key"]: entry for entry in self.manifest["objects"]}
            buffer_specs = {buffer.name: buffer for buffer in self.prim_func.buffer_map.values() if buffer.name != "__comm_descriptor"}
            backing_arrays = []
            for rank, values in arrays.items():
                for key, value in values.items():
                    if not isinstance(value, np.ndarray) or (key not in signal_keys and not value.flags.c_contiguous):
                        fail(f"{key} rank {rank} requires contiguous NumPy storage")
                    if key not in buffer_specs:
                        fail(f"unknown launch buffer {key}")
                    buffer = buffer_specs[key]
                    storage_dtype = np.dtype("uint16" if buffer.dtype == "bfloat16" else str(buffer.dtype)).newbyteorder("<")
                    if value.dtype != storage_dtype or value.shape != tuple(int(x) for x in buffer.shape):
                        fail(f"{key} rank {rank} storage dtype/shape does not match its TIR Buffer")
                    if key in object_specs:
                        expected_strides = tuple(s * value.itemsize for s in object_specs[key]["strides"])
                        if value.strides != expected_strides or value.ctypes.data % 64:
                            fail(f"{key} rank {rank} storage violates symmetric strides or 64-byte alignment")
                    if any(np.shares_memory(value, other) for other in backing_arrays):
                        fail(f"aliased allocations for {key} rank {rank} are unsupported; allocate independent symmetric objects")
                    backing_arrays.append(value)
            occupied = {}
            for region in all_regions.values():
                array = np.asarray(arrays[region.rank][region.key])
                if region.key in signal_keys:
                    simulator.register_signal(model.SignalRef(region.rank, region.key, region.offset), int(array.flat[region.offset]))
                    continue
                identity = (region.rank, region.key)
                for previous in occupied.setdefault(identity, []):
                    if max(region.offset, previous.offset) < min(region.offset + region.count, previous.offset + previous.count):
                        fail(f"partially overlapping model regions for {region.key}; use identical or disjoint tiles")
                occupied[identity].append(region)
                simulator.register_buffer(ref(region), array.reshape(-1)[region.offset : region.offset + region.count].copy())
            inputs_dir = directory / "inputs"
            inputs_dir.mkdir()
            report["input_sha256"] = {}
            for rank, values in arrays.items():
                for key, value in values.items():
                    path = inputs_dir / f"rank{rank}_{key}.bin"
                    np.ascontiguousarray(value).tofile(path)
                    report["input_sha256"][path.name] = _sha(path)

            def program(rank):
                events = {}
                for index, instruction in enumerate(plans[rank]):
                    kind, *args = instruction
                    if kind == "copy_async":
                        events[args[0]] = yield model.CopyAsync(ref(args[1]), ref(args[2]))
                    elif kind == "put_signal":
                        events[args[0]] = yield model.PutSignal(ref(args[1]), ref(args[2]), args[3], args[4])
                    elif kind == "wait":
                        yield model.Wait(events[args[0]], completion=args[1])
                    elif kind == "signal":
                        yield model.Signal(args[0], args[1], release_regions=tuple(ref(x) for x in args[2]))
                    elif kind == "wait_signal":
                        yield model.WaitSignal(args[0], args[1], covers=tuple(ref(x) for x in args[2]))
                        if args[2] and rank == slow_rank and consumer_delay:
                            yield model.Delay(consumer_delay)
                    elif kind in ("fence", "quiet"):
                        yield model.Quiet()  # Legal conservative fence lowering.
                    elif kind == "local_copy":
                        source, destination = args
                        event = yield model.Compute("local_copy", (ref(source),), (ref(destination),), lambda a: (a.copy(),))
                        yield model.Wait(event)
                    elif kind == "gemm":
                        sources, destination, body, allocations, input_regions, output_region, annotations = args

                        def compute(
                            a,
                            b,
                            sources=sources,
                            destination=destination,
                            body=body,
                            allocations=allocations,
                            input_regions=input_regions,
                            output_region=output_region,
                            annotations=annotations,
                            index=index,
                        ):
                            from .segments import compile_segment, extract_segment

                            # Check the exact mathematical contract in both modes.
                            segment = extract_segment(body, allocations, input_regions, output_region, annotations=annotations)
                            if self.mode == "protocol":
                                # BF16 bit decode; full-math oracle is external to this adapter.
                                def decode(bits):
                                    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(64, 64)

                                return ((decode(a) @ decode(b)).reshape(-1),)
                            # Cache by extracted canonical local TIR, not runtime addresses/rank/epoch.
                            digest = hashlib.sha256(tvm.ir.save_json(segment).encode()).hexdigest()
                            if digest not in compiled:
                                build_dir = directory / f"compiled_{len(compiled)}"
                                metadata = compile_segment(
                                    body,
                                    allocations,
                                    input_regions,
                                    output_region,
                                    build_dir,
                                    profile=self.compile_profile,
                                    annotations=annotations,
                                )
                                compiled[digest] = metadata
                                report["compiled_segments"].append(metadata)
                                worker_dir = directory / f"cann_{len(workers)}"
                                worker_dir.mkdir()
                                workers[digest] = _Worker(
                                    self.helper, metadata["kernel"], worker_dir, metadata["expected_cores"], segment_timeout
                                )
                            response = workers[digest].execute(rank, len(report["segments"]), a, b)
                            if "manifest" in response:
                                segment_report = response["manifest"]
                                segment_report["dsl_context"] = dict(
                                    rank=rank,
                                    instruction=index,
                                    inputs=[dict(rank=r.rank, object=r.key, offset=r.offset, count=r.count) for r in sources],
                                    output=dict(object=destination.key, offset=destination.offset, count=destination.count),
                                )
                                _json(segment_report["manifest_path"], segment_report)
                                report["segments"].append(segment_report)
                            if not response.get("passed"):
                                raise RuntimeError(response.get("error", "CANN segment failed"))
                            return (np.fromfile(response["manifest"]["output"], dtype="<f4"),)

                        event = yield model.Compute(f"gemm_{index}", tuple(ref(x) for x in sources), (ref(destination),), compute)
                        yield model.Wait(event)
                yield model.Quiet()

            simulator.run({rank: program(rank) for rank in arrays})
            for region in all_regions.values():
                if region.key not in signal_keys:
                    arrays[region.rank][region.key].reshape(-1)[region.offset : region.offset + region.count] = simulator.read_buffer(
                        ref(region)
                    )
            # Signal values persist across launches, preserving the epoch namespace.
            for event in simulator.trace_events:
                if event["name"] == "signal_arrive":
                    destination = event["args"]["destination"]
                    for region in all_regions.values():
                        if region.key in signal_keys and str(model.SignalRef(region.rank, region.key, region.offset)) == destination:
                            arrays[region.rank][region.key].flat[region.offset] = event["args"]["epoch"]
            report["passed"] = True
        except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            for worker in workers.values():
                worker.close()
            report["protocol"] = simulator.report()
            report["kernel_executed"] = any(s.get("passed") for s in report["segments"])
            _json(directory / "protocol.json", report["protocol"])
            _json(
                directory / "protocol_trace.json",
                dict(traceEvents=simulator.trace_events, metadata=dict(time_unit="logical_tick", physical_time=False)),
            )
            _json(
                directory / "trace_index.json",
                dict(
                    physical_time_aligned=False,
                    segments=[
                        {k: s[k] for k in ("rank", "segment_id", "dsl_context", "manifest_path", "pipeline_traces")}
                        for s in report["segments"]
                    ],
                ),
            )
            _json(directory / "manifest.json", report)
        return report


def compile_program(program, **kwargs):
    return CompiledProgram(program, **kwargs)
