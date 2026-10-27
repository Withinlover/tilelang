"""Bounded single-kernel SHMEM lowering using ordinary Ascend scheduling.

The host resolves only scalar protocol control. Communication, waits, local
copies and arithmetic remain in one device kernel. The communication intrinsic
owns its scratch until its synchronous completion; explicit regions connect
acquire and release to the ordinary DMA dependency graph.
"""

from collections import OrderedDict
import functools
import hashlib
import inspect
import math
from pathlib import Path

import tilelang
from tilelang import tvm
from tvm import tirx

from tilelang.analysis.distributed_checker import fail, op_name, tile_region
from tilelang.distributed.abi import P0A_PROFILE, P0B_PROFILE, manifest_digest
from .native import NativeCompiledProgram, _digest, _json, describe_instruction
from .plan import plan_program
from .transport import SCRATCH_BYTES, shmem_include_flags


MAX_FUSED_INSTRUCTIONS = 4096


def validate_vector_phase(body, allocations, inputs, output):
    """Validate the initial pure-AIV surface before scheduler transformations."""
    if len(inputs) != 1:
        fail("P0-A vector phases require exactly one GM input and one GM output")
    if any(buffer.scope() not in ("shared", "shared.dyn") for buffer in allocations):
        fail("P0-A vector phases support only UB local allocations")
    if any(buffer.dtype != "float32" for buffer in [*allocations, inputs[0].buffer, output.buffer]):
        fail("P0-A vector computation currently supports float32 only")
    if sum(math.prod(int(x) for x in b.shape) * 4 for b in allocations) + SCRATCH_BYTES > 128 * 1024:
        fail("P0-A vector UB allocations plus communication scratch exceed the 128 KiB supported budget")
    from .segments import _verify_compact_buffer

    for index, buffer in enumerate(allocations):
        _verify_compact_buffer(buffer, f"P0-A UB {buffer.name}")
        if any(buffer.data.same_as(other.data) for other in allocations[:index]):
            fail("P0-A requires independent UB allocations; aliased local views are unsupported")
    for region in [*inputs, output]:
        _verify_compact_buffer(region.buffer, f"P0-A GM {region.buffer.name}")
    allowed = {"tl.region", "tl.tileop.ascend_copy", "tl.access_ptr", "tirx.tvm_access_ptr"}
    allowed.update({"tl.simd.vld", "tl.simd.vadd", "tl.simd.vsts", "tl.simd.pset"})
    # SIMD op names are checked explicitly; unsupported math needs its own
    # tested contract before entering this small sequential profile.
    calls = []
    for statement in body:
        tirx.stmt_functor.post_order_visit(statement, lambda n: calls.append(n) if isinstance(n, tirx.Call) else None)
    for call in calls:
        name = op_name(call)
        if name == "tl.simd.pset" and (int(call.args[0]) != 32 or call.args[1].value != "PAT_ALL"):
            fail("P0-A float32 SIMD requires pset(32, PAT_ALL); the argument is element width, not lane count")
        if name not in allowed:
            fail(f"unsupported P0-A vector operation {name}; supported arithmetic is float32 SIMD vadd")
    for statement in body:
        if isinstance(statement, tirx.Evaluate):
            call = statement.value
            if op_name(call) != "tl.tileop.ascend_copy" or call.annotations:
                fail("P0-A vector phases require default GM/UB copies")
            src, dst = [tile_region(arg) for arg in call.args]
            if (
                src is None
                or dst is None
                or {src.buffer.scope().replace("shared.dyn", "shared"), dst.buffer.scope().replace("shared.dyn", "shared")}
                != {"global", "shared"}
            ):
                fail("P0-A vector phases require GM-to-UB or UB-to-GM copies")
        elif not isinstance(statement, tirx.SBlock) or statement.name_hint != "SIMD_VF":
            fail("P0-A vector phases only accept copies and explicit SIMD VF blocks")
        else:
            _verify_vector_memory(statement, allocations)


def _verify_vector_memory(block, allocations):
    """Do not let a VF hide a GM read or an untracked local alias from TD."""
    analyzer = tvm.arith.Analyzer()
    pointer_loads = []

    def walk(node):
        if isinstance(node, tirx.For):
            if node.kind != tirx.ForKind.SERIAL or node.annotations:
                fail("P0-A SIMD supports plain serial lane-group loops")
            analyzer.bind(node.loop_var, tvm.ir.Range.from_min_extent(node.min, node.extent))
        if isinstance(node, tirx.BufferStore):
            fail("P0-A SIMD scalar stores are unsupported; use vsts to declared UB")
        if isinstance(node, tirx.BufferLoad):
            fail("P0-A SIMD direct/GM loads are unsupported; use vld from declared UB")
        if isinstance(node, tirx.Call) and op_name(node) == "tl.access_ptr":
            load, count, mask = node.args
            if not isinstance(load, tirx.BufferLoad) or not any(load.buffer.same_as(buffer) for buffer in allocations):
                fail("P0-A SIMD pointers must refer to declared UB buffers")
            if not isinstance(count, tirx.IntImm) or int(count) != 64 or int(mask) not in (1, 2):
                fail("P0-A SIMD pointers require 64 float32 lanes and explicit read/write access")
            pointer_loads.append(load)
            offset = tirx.const(0, "int32")
            for index, extent in zip(load.indices, load.buffer.shape):
                if not analyzer.can_prove(index >= 0) or not analyzer.can_prove(index < extent):
                    fail("P0-A SIMD access cannot be proven within its UB allocation")
                offset = offset * extent + index
            if not analyzer.can_prove(offset + count <= math.prod(int(x) for x in load.buffer.shape)):
                fail("P0-A SIMD vector access exceeds its UB allocation")
            return
        if isinstance(node, tirx.SBlock):
            if node.alloc_buffers or node.match_buffers or node.init is not None or node.iter_vars:
                fail("P0-A SIMD VF must use the enclosing declared UB allocations")
            walk(node.body)
        elif isinstance(node, tirx.SeqStmt):
            for item in node.seq:
                walk(item)
        elif isinstance(node, (tirx.For, tirx.AttrStmt)):
            if isinstance(node, tirx.AttrStmt) and node.attr_key != "tl.simdvf_scope":
                fail(f"unsupported P0-A SIMD attribute {node.attr_key}")
            walk(node.body)
        elif isinstance(node, (tirx.Evaluate, tirx.Bind)):
            walk(node.value)
        elif isinstance(node, tirx.Call):
            for arg in node.args:
                walk(arg)
        elif isinstance(node, (tirx.IfThenElse, tirx.While)):
            fail("P0-A SIMD supports unconditional vector blocks")

    walk(block)
    loads = []
    tirx.stmt_functor.post_order_visit(block, lambda node: loads.append(node) if isinstance(node, tirx.BufferLoad) else None)
    if any(not any(load.same_as(pointer) for pointer in pointer_loads) for load in loads):
        fail("P0-A SIMD direct/GM loads are unsupported; use vld from declared UB")


def _normal_plan(plan, epoch_base):
    result = []
    for instruction in plan:
        values = list(instruction)
        if values[0] == "put_signal":
            values[5] -= epoch_base
        elif values[0] in ("signal", "wait_signal"):
            values[2] -= epoch_base
        result.append(tuple(values))
    return result


def _clone_containers(value):
    # TIR nodes and frozen Region/SignalRef values can be shared. Never expose
    # mutable Python containers owned by a cached validation result.
    if isinstance(value, dict):
        return {key: _clone_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_containers(item) for item in value)
    return value


def _container_key(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _container_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_container_key(item) for item in value)
    return value


def _remember(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _description(instruction):
    if instruction[0] == "vector":
        return {
            "op": "vector",
            "inputs": [list(x.identity) for x in instruction[1]],
            "output": list(instruction[2].identity),
            "body": [_digest(tvm.ir.save_json(x)) for x in instruction[3]],
        }
    return describe_instruction(instruction)


def _overlap(a, b):
    return a.rank == b.rank and a.key == b.key and max(a.offset, b.offset) < min(a.offset + a.count, b.offset + b.count)


def _covers(a, b):
    return a.rank == b.rank and a.key == b.key and a.offset <= b.offset and a.offset + a.count >= b.offset + b.count


def verify_fused_protocol(plans):
    """Check finite cross-rank memory order and explicit acquire/release ranges.

    Source rank order and matching signal publications define two-component
    vector clocks. Conflicting accesses must be ordered by these clocks. This
    proves only this statically expanded protocol, not physical device progress.
    """
    clocks = {0: [0, 0], 1: [0, 0]}
    positions = {0: 0, 1: 0}
    signals, accesses, acquired = {}, [], {0: [], 1: []}
    remote_writes = [i[3] for r, p in plans.items() for i in p if i[0] in ("copy_async", "put_signal") and i[3].rank != r]

    def record(rank, region, write, clock):
        if (
            not write
            and region.rank == rank
            and any(_overlap(region, other) for other in remote_writes)
            and not any(_covers(cover, region) for cover in acquired[rank])
        ):
            fail(f"P0 fused read of {region.key} requires READY/acquire covers for the received region")
        for other, previous_write, previous_clock in accesses:
            if not (write or previous_write) or not _overlap(region, other):
                continue
            if not all(a <= b for a, b in zip(previous_clock, clock)) and not all(a <= b for a, b in zip(clock, previous_clock)):
                fail(f"P0 fused unordered access/slot reuse to {region.key}; wait for ACK before overwriting a leased region")
        accesses.append((region, write, tuple(clock)))

    while any(positions[r] < len(plans[r]) for r in (0, 1)):
        progressed = False
        for rank in (0, 1):
            if positions[rank] == len(plans[rank]):
                continue
            kind, *args = plans[rank][positions[rank]]
            clock = clocks[rank]
            if kind == "wait_signal":
                ref, value, covers = args
                publications = signals.get((ref.rank, ref.key, ref.index), [])
                candidates = [entry for entry in publications if entry[0] >= value]
                if not candidates:
                    continue
                publication = min(candidates, key=lambda entry: entry[0])
                clock[:] = [max(a, b) for a, b in zip(clock, publication[1])]
                acquired[rank].extend(covers)
            clock[rank] += 1
            if kind in ("copy_async", "put_signal"):
                _, source, destination, *tail = args
                if source.rank != rank and not any(_covers(cover, source) for cover in acquired[rank]):
                    fail(f"P0 fused get of {source.key} requires READY/acquire covering the remote source lease")
                record(rank, source, False, clock)
                record(rank, destination, True, clock)
            elif kind == "local_copy":
                record(rank, args[0], False, clock)
                record(rank, args[1], True, clock)
            elif kind in ("gemm", "vector"):
                for region in args[0]:
                    record(rank, region, False, clock)
                record(rank, args[1], True, clock)
            if kind in ("signal", "put_signal"):
                ref, value = (args[0], args[1]) if kind == "signal" else (args[3], args[4])
                if kind == "signal":
                    leases = [r for r in acquired[rank] if r.rank == rank and any(_overlap(r, w) for w in remote_writes)]
                    if any(not any(_covers(release, lease) for release in args[2]) for lease in leases):
                        fail("P0 fused ACK must release every acquired receive region; add release_regions before slot reuse")
                    for region in args[2]:
                        acquired[rank] = [cover for cover in acquired[rank] if not _overlap(cover, region)]
                signals.setdefault((ref.rank, ref.key, ref.index), []).append((value, tuple(clock)))
            positions[rank] += 1
            progressed = True
        if not progressed:
            fail("P0 fused protocol has a definite signal wait deadlock")
    for rank in (0, 1):
        if any(region.rank == rank and any(_overlap(region, other) for other in remote_writes) for region in acquired[rank]):
            fail("P0 fused receive lease requires TD.signal(..., release_regions=[received_region]) before exit")


def make_fused_prim_func(program, specs, plan, *, rank, mixed):
    """Replace TD with registered communication tasks, retaining local TIR."""
    objects = {spec["key"]: program.buffer_map[program.params[spec["param_index"]]] for spec in specs}
    buffers = []
    for spec in specs:
        original = objects[spec["key"]]
        if spec["kind"] == "signal":
            original = tirx.decl_buffer((spec["physical_nbytes"] // 4,), "uint32", name=spec["key"])
        buffers.append(original)
    objects = {spec["key"]: buffer for spec, buffer in zip(specs, buffers)}
    epoch = tirx.Var("epoch0", "int32")
    scratch = tirx.decl_buffer((SCRATCH_BYTES,), "uint8", name="td_comm_scratch", scope="shared.dyn")
    null = tirx.const(0, "int64")
    allocations, annotations, statements = [scratch], {}, []

    def ptr(buffer, offset, count, mask):
        indices = []
        for extent in reversed(buffer.shape):
            indices.insert(0, offset % int(extent))
            offset //= int(extent)
        return tirx.call_intrin("handle", tvm.ir.Op.get("tl.access_ptr"), tirx.BufferLoad(buffer, indices), count, mask)

    def region_ptr(ref, mask):
        return ptr(objects[ref.key], ref.offset, ref.count, mask)

    def signal_ptr(ref, mask):
        return ptr(objects[ref.key], ref.index * 16, 1, mask)

    def task(kind, dst=null, src=null, signal=null, nbytes=0, peer=0, value=0, effects=()):
        statements.append(
            tirx.Evaluate(
                tirx.call_intrin(
                    "int32",
                    tvm.ir.Op.get("tl.ascend_distributed_task"),
                    kind,
                    dst,
                    src,
                    signal,
                    ptr(scratch, 0, SCRATCH_BYTES, 3),
                    nbytes,
                    peer,
                    value,
                    *effects,
                    annotations={"td.task_id": len(statements)},
                )
            )
        )

    for instruction in plan:
        kind, *args = instruction
        if kind in ("gemm", "vector"):
            statements.extend(args[2])
            for allocation in args[3]:
                if not any(allocation.same_as(old) for old in allocations):
                    allocations.append(allocation)
            annotations.update(args[6])
        elif kind in ("copy_async", "put_signal", "local_copy"):
            source, destination = args if kind == "local_copy" else args[1:3]
            nbytes = source.count * tvm.DataType(objects[source.key].dtype).bits // 8
            peer = rank if kind == "local_copy" else (destination.rank if source.rank == rank else source.rank)
            number = 2 if kind == "put_signal" else (1 if source.rank != rank else 0)
            task(
                number,
                region_ptr(destination, 2),
                region_ptr(source, 1),
                signal_ptr(args[3], 2) if kind == "put_signal" else null,
                nbytes,
                peer,
                epoch + args[4] if kind == "put_signal" else 0,
            )
        elif kind in ("signal", "wait_signal"):
            ref, value, regions = args
            task(
                3 if kind == "signal" else 4,
                signal=signal_ptr(ref, 2 if kind == "signal" else 3),
                peer=ref.rank,
                value=epoch + value,
                effects=[region_ptr(r, 2) for r in regions],
            )
        elif kind in ("quiet", "fence", "wait"):
            # The communication task completes its async transfer before
            # returning. Wait/fence retain a scratch dependency and a drain.
            task(5)
        else:
            fail(f"unsupported fused instruction {kind}")
    task(5)
    body = tirx.SeqStmt(statements)
    annotations["tl.buffer_versions_map"] = {scratch.data: 1}
    if mixed:
        annotations["vector_count"] = tirx.const(1, "int32")
    block = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=allocations, annotations=annotations)
    bx = tirx.Var("bx", "int32")
    axis = tirx.IterVar(tvm.ir.Range(0, 1), bx, tirx.IterVar.ThreadIndex, "blockIdx.x")
    body = tirx.SBlockRealize([], True, block)
    if mixed:
        sid = tirx.Var("sid", "int32")
        sid_axis = tirx.IterVar(tvm.ir.Range(0, 1), sid, tirx.IterVar.ThreadIndex, "cthread")
        body = tirx.For(sid, 0, 1, tirx.ForKind.THREAD_BINDING, body, thread_binding=sid_axis)
    body = tirx.For(bx, 0, 1, tirx.ForKind.THREAD_BINDING, body, thread_binding=axis)
    body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "root", body))
    return tirx.PrimFunc([*[b.data for b in buffers], epoch], body, buffer_map={b.data: b for b in buffers}).with_attr(
        {
            "global_symbol": f"td_fused_{'p0b' if mixed else 'p0a'}_rank{rank}",
            "tir.noalias": True,
            "tl.distributed_fused": "p0b" if mixed else "p0a",
        }
    )


class FusedCompiledProgram(NativeCompiledProgram):
    """One ordinary tvm_ffi kernel per rank, using the existing Session."""

    _plan_program = staticmethod(functools.partial(plan_program, allow_vector=True))

    def __init__(self, program, *, profile=P0A_PROFILE, **kwargs):
        if profile not in (P0A_PROFILE, P0B_PROFILE):
            fail("fused execution requires an explicit P0-A or P0-B profile")
        super().__init__(program, **kwargs)
        self.profile = profile
        from tilelang.env import TILELANG_TEMPLATE_PATH

        self.helper_path = Path(TILELANG_TEMPLATE_PATH) / "tl_templates/ascend/distributed.h"
        self.helper_fingerprint = hashlib.sha256(self.helper_path.read_bytes()).hexdigest()
        self.sdk_adapter_fingerprint = _digest(inspect.getsource(adapt_cann91_fused) + _CANN91_FP32_SIMD)
        from .segments import _verify_compact_buffer

        source_buffers = []
        for spec in self.manifest["objects"]:
            buffer = program.buffer_map[program.params[spec["param_index"]]]
            if spec["kind"] == "symmetric":
                _verify_compact_buffer(buffer, "fused symmetric object " + spec["key"])
            if any(buffer.data.same_as(previous.data) for previous in source_buffers):
                fail("fused symmetric objects must use independent allocations")
            source_buffers.append(buffer)
        self.manifest.update(
            {
                "profile": profile,
                "descriptor": {"abi_version": 1, "encoding": "single-kernel-shmem-v1", "device_argument": False},
                "execution": "native-single-kernel-p0a" if profile == P0A_PROFILE else "native-single-kernel-p0b",
                "device_abi": "ordered-symmetric-objects-then-int32-epoch0-v1",
                "fusion_helper_sha256": self.helper_fingerprint,
                "sdk_adapter_sha256": self.sdk_adapter_fingerprint,
                "communication_scratch_bytes": SCRATCH_BYTES,
                "communication_event_id": 7,
                "max_instructions": MAX_FUSED_INSTRUCTIONS,
            }
        )
        self._manifest_digest = manifest_digest(self.manifest)
        self._base = 0
        self._normal_plans = None
        self._reference_cache = OrderedDict()
        self._expanded_cache = OrderedDict()
        self._description_cache = OrderedDict()

    def _description(self, instruction):
        if instruction[0] not in ("gemm", "vector"):
            return describe_instruction(instruction)
        key = _container_key(instruction)
        if key not in self._description_cache:
            _remember(self._description_cache, key, describe_instruction(instruction), 128)
        return _clone_containers(self._description_cache[key])

    def _describe_plans(self, plans):
        return {str(rank): [self._description(item) for item in plan] for rank, plan in plans.items()}

    def _scalar_key(self, values):
        from .plan import _integer

        parameters = {param.name: param.dtype for param in self.prim_func.params if param not in self.prim_func.buffer_map}
        if set(values) - parameters.keys() - {"epoch0"} or parameters.keys() - values.keys():
            # Keep the shared planner's precise missing/extra-scalar diagnostic.
            super()._expand_plans(values)
        return tuple(sorted((name, _integer(value, parameters.get(name, "int32"))) for name, value in values.items()))

    def _reference(self, values):
        key = tuple((name, value) for name, value in self._scalar_key(values) if name != "epoch0")
        if key in self._reference_cache:
            self._reference_cache.move_to_end(key)
            return self._reference_cache[key]
        values = dict(values, epoch0=0)
        reference = super()._expand_plans(values)
        if not any(i[0] in ("local_copy", "gemm", "vector", "copy_async", "put_signal") for plan in reference.values() for i in plan):
            fail("fused execution requires nonempty payload or local computation; skip zero work on the host")
        if any(len(plan) > MAX_FUSED_INSTRUCTIONS for plan in reference.values()):
            fail(f"fused static expansion supports at most {MAX_FUSED_INSTRUCTIONS} instructions per rank")
        if self.profile == P0A_PROFILE and any(i[0] == "gemm" for p in reference.values() for i in p):
            fail("P0-A is pure AIV; select P0-B for GEMM fusion")
        if self.profile == P0B_PROFILE and any(i[0] == "vector" for p in reference.values() for i in p):
            fail("P0-B currently supports copy/communication and fixed GEMM64; vector arithmetic uses P0-A")
        descriptions = self._describe_plans(reference)
        translated = super()._expand_plans(dict(values, epoch0=1))
        if self._describe_plans({r: _normal_plan(p, 1) for r, p in translated.items()}) != descriptions:
            fail("fused epoch0 may only translate signal epochs; epoch-dependent control, payload regions and computation are unsupported")
        verify_fused_protocol(reference)
        from .fused_epoch import prove_epoch_translation

        proofs = [prove_epoch_translation(self.prim_func, scalars=dict(self._scalar_key(values)), rank=r) for r in (0, 1)]
        interval = (
            None
            if any(proof is None for proof in proofs)
            else (max(proof.min_base for proof in proofs), min(proof.max_base for proof in proofs))
        )
        entry = {"plans": reference, "descriptions": descriptions, "funcs": None, "keys": None, "proof": interval}
        # A scalar specialization owns its TIR and kernels. Keep this working set
        # bounded; an evicted specialization is rebuilt on its next use.
        if len(self._reference_cache) == 4:
            _, evicted = self._reference_cache.popitem(last=False)
            for kernel_key in (evicted["keys"] or {}).values():
                self._kernels.pop(kernel_key, None)
        self._reference_cache[key] = entry
        return entry

    def _expand_plans(self, values):
        entry = self._reference(values)
        self._pending_reference = entry
        key = self._scalar_key(values)
        if key not in self._expanded_cache:
            base = values["epoch0"]
            interval = entry["proof"]
            if interval is not None and interval[0] <= base <= interval[1]:
                plans = {r: _normal_plan(plan, -base) for r, plan in entry["plans"].items()}
            else:
                # Unproven IR or a base outside the proof's intermediate-int32
                # bounds must retain the complete per-epoch planner checks.
                plans = super()._expand_plans(values)
                normal = {r: _normal_plan(p, base) for r, p in plans.items()}
                if self._describe_plans(normal) != entry["descriptions"]:
                    fail(
                        "fused epoch0 may only translate signal epochs; epoch-dependent control, payload regions and computation are unsupported"
                    )
            _remember(self._expanded_cache, key, plans, 8)
        else:
            self._expanded_cache.move_to_end(key)
        return _clone_containers(self._expanded_cache[key])

    def plan(self, *, scalars=None, epoch_base=0, **kwargs):
        self._configs()  # Revalidate the bound device helper before any launch plan.
        plans = super().plan(scalars=scalars, epoch_base=epoch_base, **kwargs)
        # Publish this specialization only after every interval/ref/progress
        # check succeeds. A failed plan must not switch a previous build token
        # to a different scalar specialization.
        self._active_reference = self._pending_reference
        self._base = epoch_base
        self._normal_plans = self._active_reference["plans"]
        self._current_plan_token = _container_key(plans)
        return plans

    def _funcs(self):
        entry = self._active_reference
        if entry["funcs"] is None:
            entry["funcs"] = {
                r: make_fused_prim_func(
                    self.prim_func, self.manifest["objects"], entry["plans"][r], rank=r, mixed=self.profile == P0B_PROFILE
                )
                for r in (0, 1)
            }
        return dict(entry["funcs"])

    def _configs(self, shmem_root=None):
        if hashlib.sha256(self.helper_path.read_bytes()).hexdigest() != self.helper_fingerprint:
            fail("fusion device helper changed after compile; create a new compiled program and Session")
        configs = {"tl.enable_auto_schedule": True}
        if shmem_root is not None:
            configs["tl.device_compile_flags"] = [
                *shmem_include_flags(shmem_root),
                "-DTL_FUSION_HELPER_DIGEST_" + self.helper_fingerprint + "=1",
                "-DTL_FUSION_SDK_DIGEST_" + self.sdk_adapter_fingerprint + "=1",
            ]
        if self.compile_profile != "native" or self.transport_compile_profile != "native":
            configs["tl.ascend_sdk_compatibility"] = "cann-9.1-fused"
        return configs

    def lower(self, *, output_dir=None, scalars=None, epochs=None, shmem_root=None):
        plans = self.plan(scalars=scalars, epoch_end=epochs)
        out = Path(output_dir or self.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for rank, func in self._funcs().items():
            with tilelang.transform.PassContext(config={"tl.enable_auto_schedule": True}):
                source = tilelang.lower(func, target={"kind": "ascend", "arch": "dav-3510"}).kernel_source
            (out / f"fused_rank{rank}.asc").write_text(source)
            (out / f"fused_rank{rank}.tir").write_text(func.script(show_meta=True))
            artifacts.append({"kind": "fused", "rank": rank, "source": f"fused_rank{rank}.asc", "sha256": _digest(source)})
        result = {
            "profile": self.profile,
            "device_executed": False,
            "execution": self.manifest["execution"],
            "plans": {str(r): [_description(i) for i in p] for r, p in plans.items()},
            "artifacts": artifacts,
        }
        _json(out / "abi.json", self.manifest)
        _json(out / "lowering.json", result)
        return result

    def build(self, *, shmem_root, scalars=None, epochs=None, plans=None, output_dir=None):
        if plans is None:
            self.plan(scalars=scalars, epoch_end=epochs)
        elif self._normal_plans is None or _container_key(plans) != self._current_plan_token:
            fail("fused build plans must come from this program's validated current plan")
        self._configs()
        if tvm.ir.save_json(self.prim_func) != self.ir_json or manifest_digest(self.manifest) != self._manifest_digest:
            fail("compiled distributed source or execution manifest was modified")
        fingerprint = self._header_fingerprint(shmem_root)
        if self.header_fingerprint is not None and fingerprint != self.header_fingerprint:
            fail("SHMEM device headers changed after native compile; create a new compiled program and Session")
        artifacts = []
        entry = self._active_reference
        identity = (fingerprint, self.compile_profile, self.transport_compile_profile)
        if entry["keys"] is None or entry.get("build_identity") != identity:
            for old_key in (entry["keys"] or {}).values():
                self._kernels.pop(old_key, None)
            entry["build_identity"] = identity
            entry["keys"] = {
                rank: ("fused", rank, _digest(tvm.ir.save_json(func)), fingerprint, self.compile_profile, self.transport_compile_profile)
                for rank, func in self._funcs().items()
            }
        self._rank_keys = dict(entry["keys"])
        for rank, func in self._funcs().items():
            key = self._rank_keys[rank]
            if key not in self._kernels:
                self._kernels[key] = tilelang.compile(
                    func, target={"kind": "ascend", "arch": "dav-3510"}, execution_backend="tvm_ffi", pass_configs=self._configs(shmem_root)
                )
            if output_dir is not None:
                out = Path(output_dir).expanduser().resolve()
                out.mkdir(parents=True, exist_ok=True)
                kernel = self._kernels[key]
                source = kernel.get_kernel_source()
                (out / f"fused_rank{rank}.generated.asc").write_text(source)
                compiled = adapt_cann91_fused(source) if "tl.ascend_sdk_compatibility" in self._configs() else source
                (out / f"fused_rank{rank}.compiled.asc").write_text(compiled)
                (out / f"fused_rank{rank}.tir").write_text(func.script(show_meta=True))
                library = out / f"fused_rank{rank}.so"
                kernel.export_library(str(library))
                artifacts.append(
                    {
                        "rank": rank,
                        "library": library.name,
                        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                        "compiled_source_sha256": _digest(compiled),
                    }
                )
        self.build_report = {
            "profile": self.profile,
            "device_executed": False,
            "physical_kernel_count_per_rank": 1,
            "source_sha256": self.source_sha256,
            "transport_header_sha256": fingerprint,
            "artifacts": artifacts,
        }
        if output_dir is not None:
            _json(Path(output_dir) / "build.json", self.build_report)
        return self._kernels

    def _execute(self, plans, session, bound, report):
        values = []
        for spec in self.manifest["objects"]:
            values.append(
                bound[spec["key"]].view(shape=(spec["physical_nbytes"] // 4,), dtype="uint32")
                if spec["kind"] == "signal"
                else bound[spec["key"]].view(shape=tuple(spec["shape"]))
            )
        values.append(report["epoch_base"])
        session.launch_kernel(self._kernels[self._rank_keys[session.rank]], *values)
        report.update(device_executed=True, kernel_executed=True, physical_kernel_count=1)
        report["phases"] = [{"index": index, "op": item[0]} for index, item in enumerate(plans[session.rank])]


def adapt_cann91_fused(source):
    """Translate the fixed fused kernel SDK surface; preserve every operation."""
    import re

    preamble = "#include <tl_templates/ascend/common.h>\n#include <tl_templates/ascend/debug.h>\n"
    if source.count(preamble) != 1 or "tl_templates/ascend/distributed.h" not in source:
        raise ValueError("cann-9.1-fused requires a generated distributed kernel")
    source = source.replace(preamble, '#include "kernel_operator.h"\n#include <c_api/asc_simd.h>\n')
    source = source.replace(
        "asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0)", "asc_set_gm2l1_nz_para((64ULL << 32) | (1ULL << 16) | 1ULL)"
    )
    source = source.replace("static_cast<asc_load_l2_cache_mode>(0)", "0")
    source = source.replace("static_cast<asc_store_l2_cache_mode>(4)", "4")
    simd_include = "#include <tl_templates/ascend/simd_inst.h>\n"
    if simd_include in source:
        operations = set(re.findall(r"simd_inst::(\w+)", source))
        if operations - {"vlds_norm", "vadd", "vsts_norm"}:
            raise ValueError("cann-9.1-fused supports only FP32 vld/vadd/vsts SIMD")
        if "asc_create_mask_b64" in source:
            raise ValueError("FP32 SIMD mask width must be 32 bits")
        source = source.replace(simd_include, _CANN91_FP32_SIMD)

    source = source.replace("asc_set_l0c_copy_nz_para(1, 0, 0)", "asc_set_l0c2gm_nz2nd(1, 0, 0)")
    source = source.replace(
        "static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0",
        "0, 0, 0, 0, 0, false, true, 0, 0, false, 0, false, false, false, false",
    )
    if re.search(r"asc_set_gm2l1_nz_para\([^\n]*static_cast", source):
        raise ValueError("cann-9.1-fused supports only the fixed GEMM64 layout")
    return source


# CANN 9.1 lacks unrelated FP8/pack APIs in the general SIMD template. These
# three adapters preserve the generated FP32 VF body and its exact operands.
_CANN91_FP32_SIMD = r"""
#include <type_traits>
namespace simd_inst {
template <typename T>
__simd_callee__ inline vector_f32 vlds_norm(__ubuf__ T* src, int32_t offset) {
  static_assert(std::is_same<T, float>::value, "P0-A requires float32");
  vector_f32 dst;
  asc_loadalign(dst, src, offset);
  return dst;
}
template <typename Mode>
__simd_callee__ inline vector_f32 vadd(vector_f32 a, vector_f32 b,
                                      vector_bool mask, Mode) {
  vector_f32 dst;
  asc_add(dst, a, b, mask);
  return dst;
}
__simd_callee__ inline void vsts_norm(vector_f32 data, __ubuf__ float* base,
                                     int32_t offset, vector_bool mask) {
  asc_storealign(base, data, offset, mask);
}
}
"""
