"""Explicit native SHMEM phases compiled from the canonical P0 source program.

This profile launches independent one-AIV communication and pure-Cube compute
kernels on one stream per rank. Each communication kernel drains its own work
before exit; no pipe state or asynchronous request escapes into another kernel.
The original TD function is never passed through the ordinary native gate.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import tilelang
from tilelang import tvm

from tilelang.analysis.distributed_checker import check_distributed, fail, manifest
from tilelang.distributed.abi import MAX_EPOCH, NATIVE_PROFILE, canonical_manifest, manifest_digest
from .plan import Region, SignalRef, plan_program
from .segments import extract_segment


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _segment(instruction):
    _, _, _, body, allocations, inputs, output, annotations = instruction
    return extract_segment(body, allocations, inputs, output, annotations=annotations)


def _segment_key(instruction):
    return _digest(tvm.ir.save_json(_segment(instruction)))


def describe_instruction(instruction):
    """Address-free evidence for a concrete rank/epoch plan."""
    kind, *args = instruction
    if kind == "vector":
        return {
            "op": kind,
            "inputs": [asdict(x) for x in args[0]],
            "output": asdict(args[1]),
            "body_sha256": [_digest(tvm.ir.save_json(x)) for x in args[2]],
        }
    if kind == "gemm":
        return {"op": kind, "inputs": [asdict(x) for x in args[0]], "output": asdict(args[1]), "segment_sha256": _segment_key(instruction)}

    def encode(value):
        if isinstance(value, (Region, SignalRef)):
            return asdict(value)
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        return value

    return {"op": kind, "args": [encode(value) for value in args]}


class NativeCompiledProgram:
    """A validated distributed source with reusable ordinary native subkernels.

    Construction and ``plan`` do not load SHMEM, CANN or cannsim-helper. ``lower``
    emits device sources without a device; ``build`` uses the existing tvm_ffi
    compiler/cache; only ``run`` requires an initialized native Session.
    """

    _supports_artifact = False

    def __init__(
        self,
        program,
        *,
        profile=NATIVE_PROFILE,
        target="ascend",
        mode="native",
        compile_profile="native",
        output_dir=None,
        shmem_root=None,
        transport_compile_profile="native",
        debug=True,
        execution_policy="strict",
    ):
        if execution_policy not in ("strict", "artifact"):
            fail("execution_policy must be strict or artifact")
        if execution_policy == "artifact" and not self._supports_artifact:
            fail("artifact execution currently requires the fused P0-A or P0-B profile")
        self.execution_policy = execution_policy
        if transport_compile_profile not in ("native", "cann-9.1-shmem"):
            fail(f"unsupported transport compile profile {transport_compile_profile!r}")
        self.shmem_root = None if shmem_root is None else Path(shmem_root).expanduser().resolve()
        self.transport_compile_profile = transport_compile_profile
        self.header_fingerprint = self._header_fingerprint(self.shmem_root) if self.shmem_root is not None else None
        if profile != NATIVE_PROFILE or target != "ascend" or mode != "native":
            fail(f"native phases require profile={NATIVE_PROFILE!r}, target='ascend', mode='native'")
        if compile_profile not in ("native", "cann-9.1-gemm64"):
            fail(f"unsupported native compute compile profile {compile_profile!r}")
        check_distributed(program)
        source_manifest = manifest(program)
        if source_manifest is None:
            fail("native phases require an eager canonical TD PrimFunc")
        if len(source_manifest["groups"]) != 1 or source_manifest["groups"][0]["size"] != 2 or len(source_manifest["channels"]) != 1:
            fail("native phases require exactly two ranks, one static group and one channel")
        self.ir_json = tvm.ir.save_json(program)
        self._source_document = self.ir_json
        self._source_generation = 0
        self._identity_guard = False
        if execution_policy == "artifact":
            from .artifact_identity import source_supports_identity_guard

            self._identity_guard = source_supports_identity_guard(program, self.ir_json)
            # Own the serialized source snapshot. Public functional IR updates
            # create another root; mutable/unknown metadata retains full JSON
            # comparison on every use instead of relying on node identity.
            self._prim_func = tvm.ir.load_json(self.ir_json)
        else:
            self._prim_func = program
        self._source_root = self._prim_func
        self.source_sha256 = _digest(self.ir_json)
        self.source_manifest = canonical_manifest(source_manifest)
        self.manifest = {
            **self.source_manifest,
            "profile": NATIVE_PROFILE,
            "program_sha256": self.source_sha256,
            "source_profile": self.source_manifest["profile"],
            "source_descriptor_encoding": self.source_manifest["descriptor"]["encoding"],
            "descriptor": {"abi_version": 1, "encoding": "host-phases-shmem-v1", "device_argument": False},
            "execution": "native-sequential-host-phases",
            "execution_policy": execution_policy,
            "target_arch": "dav-3510",
            "compute_compile_profile": compile_profile,
            "transport_compile_profile": transport_compile_profile,
            "transport_header_sha256": self.header_fingerprint,
        }
        self.profile, self.mode, self.compile_profile = profile, mode, compile_profile
        self.debug = debug
        self.output_dir = Path(output_dir or "output/distributed/native").expanduser().resolve()
        self._specs = {entry["key"]: entry for entry in self.manifest["objects"]}
        if len(self._specs) != len(self.manifest["objects"]):
            fail("native phases require unique symmetric object names")
        descriptor_index = self.source_manifest["descriptor"]["param_index"]
        registered = {entry["param_index"] for entry in self.manifest["objects"]}
        for index, parameter in enumerate(program.params):
            if parameter in program.buffer_map and index != descriptor_index and index not in registered:
                fail("native phases require every data buffer to be an explicit SymBuffer parameter")
        self._manifest_digest = manifest_digest(self.manifest)
        self._kernels = {}

    @property
    def prim_func(self):
        return self._prim_func

    @prim_func.setter
    def prim_func(self, value):
        self._source_generation += 1
        self._prim_func = value

    @property
    def source_validation(self):
        return "identity" if self.execution_policy == "artifact" and self._identity_guard else "serialized"

    def _validate_source_identity(self, *, strict=False):
        if self.execution_policy != self.manifest["execution_policy"]:
            fail("compiled execution policy was modified")
        if self.ir_json != self._source_document or self.source_sha256 != self.manifest["program_sha256"]:
            fail("compiled distributed source or execution manifest was modified")
        if self.execution_policy == "artifact" and (self._source_generation or not self.prim_func.same_as(self._source_root)):
            fail("compiled artifact source was rebound; create a new compiled program")
        if (strict or self.execution_policy == "strict" or not self._identity_guard) and tvm.ir.save_json(self.prim_func) != self.ir_json:
            fail("compiled distributed source or execution manifest was modified")
        if manifest_digest(self.manifest) != self._manifest_digest:
            fail("compiled distributed source or execution manifest was modified")

    _plan_program = staticmethod(plan_program)

    @staticmethod
    def _header_fingerprint(shmem_root):
        from .transport import shmem_include_flags

        macro = shmem_include_flags(shmem_root)[-1]
        return macro.removeprefix("-DTL_SHMEM_HEADER_DIGEST_").removesuffix("=1")

    def plan(self, *, scalars=None, epoch_base=0, epoch_end=None, initial_signals=None):
        """Validate every rank and phase before any device request is submitted."""
        self._validate_source_identity()
        if isinstance(epoch_base, bool) or not isinstance(epoch_base, int) or not 0 <= epoch_base < MAX_EPOCH:
            fail("epoch base must be a nonnegative int32 below the final signal epoch")
        values = dict(scalars or {})
        if "epoch0" in values and values["epoch0"] != epoch_base:
            fail("epoch0 must equal the prepared native epoch base")
        values.setdefault("epoch0", epoch_base)
        plans = self._expand_plans(values)
        published = [
            instruction[5] if instruction[0] == "put_signal" else instruction[2]
            for plan in plans.values()
            for instruction in plan
            if instruction[0] in ("put_signal", "signal")
        ]
        if epoch_end is None:
            epoch_end = max([epoch_base + 1, *published])
        if isinstance(epoch_end, bool) or not isinstance(epoch_end, int) or not epoch_base < epoch_end <= MAX_EPOCH:
            fail("native prepared epoch interval must be positive and fit int32")
        for rank, instructions in plans.items():
            for instruction in instructions:
                kind, *args = instruction
                refs = []
                if kind in ("copy_async", "put_signal"):
                    refs.extend(args[1:3])
                    if kind == "put_signal":
                        refs.append(args[3])
                        if not epoch_base < args[4] <= epoch_end:
                            fail(f"published epoch {args[4]} is outside the prepared native interval")
                elif kind in ("signal", "wait_signal"):
                    refs.append(args[0])
                    refs.extend(args[2])
                    if kind == "signal" and not epoch_base < args[1] <= epoch_end:
                        fail(f"published epoch {args[1]} is outside the prepared native interval")
                    if kind == "wait_signal" and not 1 <= args[1] <= epoch_end:
                        fail(f"wait epoch {args[1]} is outside the prepared native interval")
                elif kind == "local_copy":
                    refs.extend(args)
                elif kind in ("gemm", "vector"):
                    refs.extend([*args[0], args[1]])
                for ref in refs:
                    self._validate_ref(ref)
                if kind in ("copy_async", "put_signal", "local_copy"):
                    transfers = args[1:3] if kind != "local_copy" else args
                    for ref in transfers:
                        item_bytes = tvm.DataType(self._specs[ref.key]["dtype"]).bits // 8
                        if ref.offset * item_bytes % 32 or ref.count * item_bytes % 32 or ref.count * item_bytes >= 2**32:
                            fail("native MTE regions require 32-byte aligned offsets and positive 32-byte-multiple lengths below 2**32")
                if kind == "local_copy":
                    source, destination = args
                    if source.key == destination.key and max(source.offset, destination.offset) < min(
                        source.offset + source.count, destination.offset + destination.count
                    ):
                        fail("native local_copy requires disjoint source and destination regions; overlapping copies are unsupported")
                if kind in ("copy_async", "put_signal"):
                    source, destination = args[1:3]
                    if source.rank == destination.rank or rank not in (source.rank, destination.rank):
                        fail("native transfers require exactly one local endpoint and the other rank")
                    if self._specs[source.key]["dtype"] != self._specs[destination.key]["dtype"] or source.count != destination.count:
                        fail("native transfer endpoints require equal dtype and count")
        self._check_progress(plans, epoch_base, initial_signals=initial_signals)
        return plans

    def _expand_plans(self, values):
        return {rank: self._plan_program(self.prim_func, rank=rank, scalars=values, world_size=2)[0] for rank in (0, 1)}

    def _describe_plans(self, plans):
        return {str(rank): [describe_instruction(item) for item in plan] for rank, plan in plans.items()}

    @staticmethod
    def _check_progress(plans, epoch_base, *, initial_signals=None):
        """Reject a proven wait cycle in this finite, conservatively drained plan.

        Previous launches can only have published values <= epoch_base. Taking
        that upper bound as every initial signal is optimistic: a stalled plan
        cannot be rescued by an actual older value. A successful traversal is
        not a proof of payload leases, data races or physical transport progress.
        A managed binding supplies its known signal state instead, including
        zero-valued cells in a newly allocated workspace. Such cells cannot
        inherit another binding's publications just because epochs advanced.
        """
        positions = {rank: 0 for rank in plans}
        signals = {} if initial_signals is None else dict(initial_signals)
        initial_value = epoch_base if initial_signals is None else 0
        while any(positions[rank] < len(plan) for rank, plan in plans.items()):
            progressed = False
            waiting = []
            for rank, instructions in plans.items():
                index = positions[rank]
                if index == len(instructions):
                    continue
                kind, *args = instructions[index]
                if kind == "wait_signal":
                    ref, expected, _ = args
                    observed = signals.get((ref.rank, ref.key, ref.index), initial_value)
                    if observed < expected:
                        waiting.append(f"rank {rank} waits {ref.key}[{ref.index}] >= {expected}, published <= {observed}")
                        continue
                if kind in ("signal", "put_signal"):
                    ref, value = (args[0], args[1]) if kind == "signal" else (args[3], args[4])
                    key = (ref.rank, ref.key, ref.index)
                    previous = signals.get(key, initial_value)
                    if value < previous:
                        fail(f"native sequential signal values must not decrease for {ref.key}[{ref.index}]")
                    signals[key] = value
                positions[rank] += 1
                progressed = True
            if not progressed:
                fail("native finite plan has a definite signal wait deadlock: " + "; ".join(waiting))
        return signals

    def _validate_ref(self, ref):
        if ref.rank not in (0, 1) or ref.key not in self._specs:
            fail("native phase references an unknown peer or symmetric object")
        spec = self._specs[ref.key]
        size = math.prod(spec["shape"])
        if isinstance(ref, SignalRef):
            if spec["kind"] != "signal" or not 0 <= ref.index < size:
                fail("native signal reference must select one declared 64-byte cell")
        elif spec["kind"] != "symmetric" or ref.offset < 0 or ref.count <= 0 or ref.offset + ref.count > size:
            fail("native data region exceeds its declared symmetric allocation")

    def _transport_spec(self, instruction, rank):
        kind, *args = instruction
        if kind in ("copy_async", "put_signal"):
            source, destination = args[1:3]
            nbytes = source.count * tvm.DataType(self._specs[source.key]["dtype"]).bits // 8
            direction = "put" if source.rank == rank else "get"
            return ("put_signal" if kind == "put_signal" else direction, nbytes)
        if kind in ("signal", "wait_signal"):
            return kind, 0
        if kind == "local_copy":
            return "local_copy", args[0].count * tvm.DataType(self._specs[args[0].key]["dtype"]).bits // 8
        return None

    def _unique(self, plans):
        transports, segments = {}, {}
        for rank, instructions in plans.items():
            for instruction in instructions:
                spec = self._transport_spec(instruction, rank)
                if spec is not None:
                    transports[spec] = None
                elif instruction[0] == "gemm":
                    segments[_segment_key(instruction)] = _segment(instruction)
        return transports, segments

    def lower(self, *, output_dir=None, scalars=None, epochs=None, shmem_root=None):
        """Write address-free plans and generated sources; never initialize a device."""
        plans = self.plan(scalars=scalars, epoch_end=epochs)
        out = Path(output_dir or self.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        from .transport import lower_transport

        transports, segments = self._unique(plans)
        artifacts = []
        for kind, nbytes in transports:
            source = lower_transport(kind, nbytes=nbytes, shmem_root=shmem_root)
            path = out / f"transport_{kind}_{nbytes}.asc"
            path.write_text(source)
            artifacts.append({"kind": kind, "nbytes": nbytes, "source": path.name, "sha256": _digest(source)})
        for digest, segment in segments.items():
            source = tilelang.lower(segment, target={"kind": "ascend", "arch": "dav-3510"}).kernel_source
            path = out / f"gemm_{digest}.asc"
            path.write_text(source)
            (out / f"gemm_{digest}.tir").write_text(segment.script(show_meta=True))
            artifacts.append({"kind": "gemm", "segment_sha256": digest, "source": path.name, "sha256": _digest(source)})
        result = {
            "profile": self.profile,
            "source_sha256": self.source_sha256,
            "compile_profile": self.compile_profile,
            "device_executed": False,
            "plans": {str(rank): [describe_instruction(item) for item in plan] for rank, plan in plans.items()},
            "artifacts": artifacts,
        }
        (out / "program.tir").write_text(self.prim_func.script(show_meta=True))
        (out / "program.json").write_text(self.ir_json)
        _json(out / "abi.json", self.manifest)
        _json(out / "lowering.json", result)
        return result

    def build(self, *, shmem_root, scalars=None, epochs=None, plans=None, output_dir=None):
        """Compile native subkernels through the normal tvm_ffi cache, without launch."""
        from .transport import build_transport

        plans = plans if plans is not None else self.plan(scalars=scalars, epoch_end=epochs)
        fingerprint = self._header_fingerprint(shmem_root)
        if self.header_fingerprint is not None and fingerprint != self.header_fingerprint:
            fail("SHMEM device headers changed after native compile; create a new compiled program and Session")
        transports, segments = self._unique(plans)
        for kind, nbytes in transports:
            key = ("transport", kind, nbytes, fingerprint, self.transport_compile_profile)
            if key not in self._kernels:
                self._kernels[key] = build_transport(
                    kind, nbytes=nbytes, shmem_root=shmem_root, output_dir=output_dir, sdk_compatibility=self.transport_compile_profile
                )
        for digest, segment in segments.items():
            key = ("gemm", digest, self.compile_profile)
            if key not in self._kernels:
                configs = {} if self.compile_profile == "native" else {"tl.ascend_sdk_compatibility": self.compile_profile}
                self._kernels[key] = tilelang.compile(
                    segment, target={"kind": "ascend", "arch": "dav-3510"}, execution_backend="tvm_ffi", pass_configs=configs
                )
        if output_dir is not None:
            out = Path(output_dir).expanduser().resolve()
            out.mkdir(parents=True, exist_ok=True)
            artifacts = []
            for key, kernel in self._kernels.items():
                name = "_".join(str(part) for part in key[:3])
                source = kernel.get_kernel_source()
                (out / f"{name}.generated.asc").write_text(source)
                compiled_source = source
                if key[0] == "gemm" and self.compile_profile == "cann-9.1-gemm64":
                    from .segments import adapt_cann91_gemm64

                    compiled_source = adapt_cann91_gemm64(source)
                elif key[0] == "transport" and self.transport_compile_profile == "cann-9.1-shmem":
                    from .transport import adapt_cann91_shmem

                    compiled_source = adapt_cann91_shmem(source)
                (out / f"{name}.compiled.asc").write_text(compiled_source)
                (out / f"{name}.tir").write_text(kernel.prim_func.script(show_meta=True))
                library = out / f"{name}.so"
                kernel.export_library(str(library))
                artifacts.append(
                    {
                        "key": list(key),
                        "library": library.name,
                        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                        "compiled_source_sha256": _digest(compiled_source),
                    }
                )
            self.build_report = {
                "profile": self.profile,
                "source_sha256": self.source_sha256,
                "transport_header_sha256": fingerprint,
                "device_executed": False,
                "artifacts": artifacts,
            }
            _json(out / "build.json", self.build_report)
        return self._kernels

    def _prepare_executable(self, *, shmem_root, plans, output_dir=None):
        self.build(shmem_root=shmem_root, plans=plans, output_dir=output_dir)
        return None

    def run(self, *, session, ticket, buffers=None, scalars=None, **kwargs):
        started = time.perf_counter()
        timings = {}
        if kwargs:
            fail(f"unsupported native launch options: {sorted(kwargs)}")
        if self.header_fingerprint is None:
            fail("native execution requires D.compile(..., shmem_root=...) to bind SHMEM header contents into its manifest")
        if session.profile != self.profile or session.world_size != 2:
            fail("native compiled profile and Session do not match")
        if manifest_digest(ticket.manifest) != self._manifest_digest or ticket.kernel is not self:
            fail("native launch ticket does not bind this compiled program")
        binding = getattr(ticket, "binding", None)
        registered = session.ticket_buffers(ticket) if hasattr(session, "ticket_buffers") else session.objects
        bound = dict(registered if buffers is None else buffers)
        if set(bound) != set(self._specs):
            fail("native buffer bindings must match exactly the compiled symmetric objects")
        timings["validate_bindings"] = time.perf_counter() - started
        stamp = time.perf_counter()
        plans = self.plan(
            scalars=scalars,
            epoch_base=ticket.base,
            epoch_end=ticket.end,
            initial_signals=None if binding is None else binding.signal_values,
        )
        timings["plan_validation"] = time.perf_counter() - stamp
        stamp = time.perf_counter()
        for key in self._specs:
            if key not in bound or bound[key] is not registered.get(key):
                fail(f"native binding {key!r} must use its Session-owned symmetric allocation")
        if session.rank not in (0, 1):
            fail("native Session rank is outside the static group")
        peers = ticket.channel.peers
        expected = {rank: (1 - rank,) for rank in range(2)}
        if peers != expected:
            fail("native channel must contain the single other rank for each participant")
        plan_json = self._describe_plans(plans)
        timings["validate_bindings"] += time.perf_counter() - stamp
        stamp = time.perf_counter()
        session.preflight(
            {
                "source_sha256": self.source_sha256,
                "manifest_digest": self._manifest_digest,
                "transport_abi": session.abi_fingerprint,
                "plans": plan_json,
            }
        )
        timings["plan_preflight"] = time.perf_counter() - stamp
        stamp = time.perf_counter()
        build_error = None
        artifact_id = None
        try:
            artifact_id = self._prepare_executable(
                shmem_root=session.shmem_root,
                plans=plans,
                output_dir=self.output_dir / f"compiled_rank{session.rank}" if self.debug else None,
            )
        except Exception as error:
            build_error = error
        timings["build_or_cache_validation"] = time.perf_counter() - stamp
        stamp = time.perf_counter()
        session.preflight(
            {"compiled": build_error is None, "error": None if build_error is None else str(build_error), "artifact_id": artifact_id}
        )
        timings["build_preflight"] = time.perf_counter() - stamp
        stamp = time.perf_counter()
        if build_error is not None:
            raise build_error
        directory = None
        if self.debug:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix=f"native_rank{session.rank}_", dir=self.output_dir))
            _json(directory / "plan.json", plan_json[str(session.rank)])
            _json(directory / "abi.json", self.manifest)
            (directory / "program.json").write_text(self.ir_json)
        report = {
            "profile": self.profile,
            "run_dir": None if directory is None else str(directory),
            "rank": session.rank,
            "source_sha256": self.source_sha256,
            "epoch_base": ticket.base,
            "epoch_end": ticket.end,
            "passed": False,
            "device_executed": False,
            "kernel_executed": False,
            "completed": False,
            "execution": self.manifest["execution"],
            "execution_policy": self.execution_policy,
            "source_validation": self.source_validation,
            "artifact_id": artifact_id,
            "transport_abi": session.abi_fingerprint,
            "phases": [],
            "errors": [],
        }

        timings["report_finalize"] = time.perf_counter() - stamp
        stamp = time.perf_counter()
        try:
            self._execute(plans, session, bound, report)
            timings["bind_and_submit"] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            session.synchronize()
            timings["stream_sync"] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            if binding is not None:
                signals = self._check_progress(plans, ticket.base, initial_signals=binding.signal_values)
                report["published_signals"] = [[*key, value] for key, value in sorted(signals.items())]
            report["passed"] = report["completed"] = True
        except BaseException as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
            raise
        finally:
            report["phases_count"] = len(report["phases"])
            timings["report_finalize"] += time.perf_counter() - stamp
            timings["total"] = time.perf_counter() - started
            report["timing_basis"] = "host_wall"
            report["timing_seconds"] = timings
            if directory is not None:
                _json(directory / "manifest.json", report)
        return report

    def _execute(self, plans, session, bound, report):
        def data_view(region, *, tile=False):
            spec = self._specs[region.key]
            byte_offset = region.offset * tvm.DataType(spec["dtype"]).bits // 8
            if tile:
                return bound[region.key].view(shape=(64, 64), byte_offset=byte_offset)
            nbytes = region.count * tvm.DataType(spec["dtype"]).bits // 8
            return bound[region.key].view(shape=(nbytes,), dtype="uint8", byte_offset=byte_offset)

        def signal_view(ref):
            return bound[ref.key].view(shape=(16,), dtype="uint32", byte_offset=ref.index * 64)

        for index, instruction in enumerate(plans[session.rank]):
            kind, *args = instruction
            spec = self._transport_spec(instruction, session.rank)
            if spec is not None:
                op, nbytes = spec
                kernel = self._kernels[("transport", op, nbytes, self.header_fingerprint, self.transport_compile_profile)]
                if kind in ("copy_async", "put_signal"):
                    source, destination = args[1:3]
                    peer = destination.rank if source.rank == session.rank else source.rank
                    values = [data_view(destination), data_view(source), peer]
                    if kind == "put_signal":
                        values.extend([signal_view(args[3]), args[4]])
                elif kind == "signal":
                    values = [signal_view(args[0]), args[0].rank, args[1]]
                elif kind == "wait_signal":
                    values = [signal_view(args[0]), args[1]]
                elif kind == "local_copy":
                    values = [data_view(args[1]), data_view(args[0]), session.rank]
                session.launch_kernel(kernel, *values)
                del values
                report["device_executed"] = report["kernel_executed"] = True
            elif kind == "gemm":
                kernel = self._kernels[("gemm", _segment_key(instruction), self.compile_profile)]
                values = [data_view(region, tile=True) for region in (*args[0], args[1])]
                session.launch_kernel(kernel, *values)
                del values
                report["device_executed"] = report["kernel_executed"] = True
            elif kind not in ("wait", "fence", "quiet"):
                fail(f"unhandled native phase {kind}")
            report["phases"].append({"index": index, "op": kind})


def compile_program(program, **kwargs):
    return NativeCompiledProgram(program, **kwargs)


def launch_native(kernel, *, ticket, buffers=None, scalars=None, stream=None, **kwargs):
    """Bind one native Session launch while retaining the ordinary Run API."""
    from tilelang.distributed.launch import Run, RunFailedError

    if not isinstance(kernel, NativeCompiledProgram):
        raise TypeError("native Session launch requires a NativeCompiledProgram")
    if stream is not None:
        fail("native phases use the Session-owned stream; external streams are unsupported")
    session = ticket.session
    session.mark_running(ticket)
    run = Run(session, ticket)
    try:
        run.result = kernel.run(session=session, ticket=ticket, buffers=buffers, scalars=scalars, **kwargs)
        if not run.result["passed"]:
            raise RunFailedError(run)
        run.finished = True
    except BaseException as error:
        session.fail(error)
        raise
    return run


def complete_native(run):
    if run.completed:
        return run.result
    if not run.finished:
        raise RuntimeError("native complete requires a successfully finished run")
    run.session.mark_complete(run.ticket)
    binding = getattr(run.ticket, "binding", None)
    if binding is not None:
        # Commit the host's signal history only after every rank has completed.
        binding.signal_values.update({tuple(entry[:3]): entry[3] for entry in run.result["published_signals"]})
    run.completed = True
    return run.result
