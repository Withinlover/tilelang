"""Supervised dense two-rank AllGather/GEMM shape sweep (real K accumulation).

This entry is independent of benchmark_p0.py's original three-round baseline.
Compilation, CPU checks and physical execution are separate stages. The local
baseline launches one grid kernel for all output tiles; P0-B uses one mixed
block. Host execution and end-to-end costs remain visible in every device run.
"""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback

if os.environ.get("P0_SHAPES_CACHE_ROOT"):
    os.environ["TILELANG_CACHE_DIR"] = str(Path(os.environ["P0_SHAPES_CACHE_ROOT"]) / f"rank{os.environ.get('RANK', 'compile')}")

from examples.distributed.benchmark_p0 import describe_samples, paired_summary, summarize_stages
from examples.distributed.example_p0_shapes import (
    allgather_dense_gemm,
    bulk_exchange,
    check_dense_output,
    local_dense_gemm,
    make_dense_inputs,
    pack_local_inputs,
    reference_output,
    tile_counts,
    unpack_output,
    workload,
)
from examples.distributed.run_p0_validation import assess_performance, run_command, write_json


def parse_shapes(value):
    shapes = []
    for item in value.split(","):
        if not re.fullmatch(r"[0-9]+(?:x[0-9]+x[0-9]+)?", item):
            raise ValueError("shapes must be comma-separated sizes or MxNxK, e.g. 128,256,1024")
        dimensions = tuple(map(int, item.split("x")))
        shape = dimensions * 3 if len(dimensions) == 1 else dimensions
        tile_counts(*shape)
        if shape in shapes:
            raise ValueError("duplicate shape")
        shapes.append(shape)
    if not 1 <= len(shapes) <= 12:
        raise ValueError("select 1..12 shapes")
    return shapes


def compile_workload(args):
    import tilelang
    import tilelang.distributed as D

    started = time.perf_counter()
    shape = (args.m, args.n, args.k)
    configs = {"tl.ascend_sdk_compatibility": "cann-9.1-packed-gemm"} if args.sdk_profile == "cann-9.1" else {}
    local = (
        tilelang.compile(
            local_dense_gemm(*shape), target={"kind": "ascend", "arch": "dav-3510"}, execution_backend="tvm_ffi", pass_configs=configs
        )
        if args.mode != "p0b"
        else None
    )
    compiled = None
    if args.mode != "hccl":
        profile = "ascend950_shmem_mte_p0a_v1" if args.mode == "p0a" else "ascend950_shmem_mte_p0b_v1"
        compiled = D.compile(
            (bulk_exchange if args.mode == "p0a" else allgather_dense_gemm)(*shape),
            profile=profile,
            mode="native",
            shmem_root=args.shmem_root,
            compile_profile="cann-9.1-gemm64" if args.sdk_profile == "cann-9.1" else "native",
            transport_compile_profile="cann-9.1-shmem" if args.sdk_profile == "cann-9.1" else "native",
            execution_policy=args.execution_policy,
            debug=False,
        )
        compiled.build(shmem_root=args.shmem_root, epochs=1, output_dir=args.output_dir / f"kernels_rank{os.environ.get('RANK', '0')}")
    return local, compiled, time.perf_counter() - started


def worker(args):
    import tilelang.distributed as D

    rank = int(os.environ.get("RANK", "0"))
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "rank": rank,
        "mode": args.mode,
        "passed": False,
        "device_executed": False,
        "execution_policy": None if args.mode == "hccl" else args.execution_policy,
        "source_validation": None,
        "launch_validation": None,
        "workload": workload(args.m, args.n, args.k),
        "checks": [],
        "timing_samples": [],
    }
    try:
        local, compiled, first_build = compile_workload(args)
        result["first_build_or_cache_load_seconds"] = first_build
        if args.stage == "compile":
            result.update(passed=True, compiled=True)
            write_json(root / "build.json", result)
            return 0
        if os.environ.get("WORLD_SIZE") != "2" or os.environ.get("RANK") not in ("0", "1"):
            raise ValueError("device workers require exactly two torchrun ranks")
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401

        device = args.devices[rank]
        torch.npu.set_device(device)
        mt, nt, kt = tile_counts(args.m, args.n, args.k)
        a_shape, b_shape, out_shape = (mt, kt, 64, 64), (nt, kt, 64, 64), (2, mt, nt, 64, 64)
        session, objects, channel = None, {}, None
        if args.mode == "hccl":
            dist.init_process_group("hccl", timeout=timedelta(seconds=args.operation_timeout))
            tensor_a = torch.empty(a_shape, dtype=torch.bfloat16, device=f"npu:{device}")
            gathered = [torch.empty_like(tensor_a) for _ in range(2)]
        else:
            session = D.init(
                profile=compiled.profile,
                rank=rank,
                device=device,
                bootstrap=f"tcp://127.0.0.1:{args.bootstrap_port}",
                control_port=args.control_port,
                shmem_root=args.shmem_root,
                shmem_library=args.shmem_library,
                timeout=args.operation_timeout,
            )
            for spec in compiled.manifest["objects"]:
                common = dict(key=spec["key"], shape=spec["shape"], dtype=spec["dtype"])
                objects[spec["key"]] = (
                    D.alloc_signals(session, **common)
                    if spec["kind"] == "signal"
                    else D.alloc_symmetric(session, remote_access=spec["remote_access"], **common)
                )
            channel = D.channel(session, key="comm")
            if args.measure_device:
                session.prepare_device_timing()
        tensor_b = torch.empty(b_shape, dtype=torch.bfloat16, device=f"npu:{device}") if local is not None else None
        tensor_out = torch.empty(out_shape, dtype=torch.float32, device=f"npu:{device}") if local is not None else None
        events = None
        if args.measure_device and local is not None:
            events = tuple(torch.npu.Event(enable_timing=True) for _ in range(3 if session is None else 2))
            # Initialize the event resources on the same current stream used by
            # HCCL and the ordinary TileLang kernel, outside measured samples.
            for event in events:
                event.record()
            torch.npu.synchronize(device)
        for iteration in range(args.warmup + args.iterations):
            originals_a, originals_b = make_dense_inputs(args.m, args.n, args.k, 1900 + iteration)
            expected = reference_output(originals_a, originals_b)
            alignment_started = time.perf_counter()
            if session is None:
                dist.barrier()
            else:
                session.preflight({"shape_benchmark_iteration": iteration})
            torch.npu.synchronize(device)
            e2e_started = time.perf_counter()
            stages, native, device_intervals = {}, {}, {}
            packed_a, packed_b = pack_local_inputs(originals_a[rank], originals_b)
            packed_ended = time.perf_counter()
            stages["pack"] = packed_ended - e2e_started
            staging_started = packed_ended
            if session is None:
                tensor_a.copy_(torch.from_numpy(packed_a).view(torch.bfloat16))
            else:
                ticket = D.prepare_launch(
                    session, compiled, channel=channel, epochs=1, protocol="p0-dense-shapes-v1", parameters={"iteration": iteration}
                )
                staging_started = time.perf_counter()
                stages["prepare"] = staging_started - packed_ended
                D.copy_in(objects["Send"], packed_a, ticket=ticket)
                if args.mode == "p0b":
                    D.copy_in(objects["B"], packed_b, ticket=ticket)
            if tensor_b is not None:
                tensor_b.copy_(torch.from_numpy(packed_b).view(torch.bfloat16))
            staging_ended = time.perf_counter()
            torch.npu.synchronize(device)
            execution_started = time.perf_counter()
            stages["staging"] = staging_ended - staging_started
            stages["staging_sync"] = execution_started - staging_ended
            if session is None:
                if events is not None:
                    events[0].record()
                dist.all_gather(gathered, tensor_a)
                if events is not None:
                    events[1].record()
                collective_ended = time.perf_counter()
                stages["collective_submit"] = collective_ended - execution_started
                local(gathered[0], gathered[1], tensor_b, tensor_out)
                if events is not None:
                    events[2].record()
                submit_ended = time.perf_counter()
                stages["local_gemm_submit"] = submit_ended - collective_ended
            else:
                run = D.launch(compiled, ticket=ticket, **({"measure_device": True} if args.measure_device else {}))
                launch_ended = time.perf_counter()
                execution = D.complete(run)
                complete_ended = time.perf_counter()
                stages["launch"], stages["complete"] = launch_ended - execution_started, complete_ended - launch_ended
                if execution.get("physical_kernel_count") != 1 or execution.get("execution_policy") != args.execution_policy:
                    raise AssertionError("expected one fused kernel with the requested validation policy")
                guard = execution.get("source_validation")
                if guard not in ("identity", "serialized") or result["source_validation"] not in (None, guard):
                    raise AssertionError("native source validation strategy is missing or changed")
                result["source_validation"] = guard
                launch_guard = execution.get("launch_validation")
                if launch_guard not in ("static_contract", "full_plan") or result["launch_validation"] not in (None, launch_guard):
                    raise AssertionError("native launch validation strategy is missing or changed")
                result["launch_validation"] = launch_guard
                if execution.get("timing_basis") != "host_wall":
                    raise AssertionError("native launch must provide host-wall attribution")
                native = execution["timing_seconds"]
                if args.measure_device:
                    duration = execution.get("device_stream_seconds")
                    if (
                        execution.get("device_timing_basis") != "acl_event_stream_interval_including_submission_gaps"
                        or type(duration) not in (int, float)
                        or not math.isfinite(duration)
                        or duration < 0
                    ):
                        raise AssertionError("missing valid timing on the native Session stream")
                    device_intervals["communication" if args.mode == "p0a" else "fused"] = duration
                submit_ended = complete_ended
                if local is not None:
                    if events is not None:
                        events[0].record()
                    views = [objects["Send" if owner == rank else "Recv"].view(shape=a_shape) for owner in range(2)]
                    local(views[0], views[1], tensor_b, tensor_out)
                    if events is not None:
                        events[1].record()
                    submit_ended = time.perf_counter()
                    stages["local_gemm_submit"] = submit_ended - complete_ended
            # P0-A must finish reading Recv before the next iteration's two-rank
            # prepare handshake permits another put to overwrite that buffer.
            torch.npu.synchronize(device)
            execution_ended = time.perf_counter()
            if session is not None and local is not None:
                del views
            packed_output = tensor_out.cpu().numpy() if tensor_out is not None else session.copy_out(objects["Output"])
            readback_ended = time.perf_counter()
            actual = unpack_output(packed_output)
            e2e_ended = time.perf_counter()
            if events is not None:
                if session is None:
                    device_intervals["collective_and_local_gemm"] = events[0].elapsed_time(events[2]) / 1000
                    device_intervals["collective"] = events[0].elapsed_time(events[1]) / 1000
                    device_intervals["local_gemm"] = events[1].elapsed_time(events[2]) / 1000
                else:
                    device_intervals["local_gemm"] = events[0].elapsed_time(events[1]) / 1000
            if any(not math.isfinite(value) or value < 0 for value in device_intervals.values()):
                raise AssertionError("device stream intervals must be finite and nonnegative")
            stages.update(
                final_sync=execution_ended - submit_ended, readback=readback_ended - execution_ended, unpack=e2e_ended - readback_ended
            )
            sample = {
                "iteration": iteration,
                "warmup": iteration < args.warmup,
                "execution_seconds": execution_ended - execution_started,
                "end_to_end_seconds": e2e_ended - e2e_started,
                "stages_seconds": stages,
                "native_run_seconds": native,
                "device_intervals_seconds": device_intervals,
                "outside_e2e_alignment_seconds": e2e_started - alignment_started,
            }
            result["timing_samples"].append(sample)
            result["device_executed"] = True
            check = {"iteration": iteration, **check_dense_output(actual, expected)}
            result["checks"].append(check)
            if not check["passed"]:
                raise AssertionError(f"iteration {iteration}: {check['mismatches']} incorrect output elements")
            if iteration == 0:
                result["first_execution_seconds"] = sample["execution_seconds"]
                result["first_end_to_end_seconds"] = sample["end_to_end_seconds"]
        steady = [sample for sample in result["timing_samples"] if not sample["warmup"]]
        result.update(
            passed=True,
            execution=describe_samples(s["execution_seconds"] for s in steady),
            end_to_end=describe_samples(s["end_to_end_seconds"] for s in steady),
            stages=summarize_stages(steady, "stages_seconds"),
            native_run=summarize_stages(steady, "native_run_seconds"),
        )
        if session is None:
            reports = [None, None]
            dist.all_gather_object(reports, result)
            dist.destroy_process_group()
        else:
            reports = session.control.all_gather(result)
            D.close(session)
        if rank == 0:
            if any(report[key] != result[key] for report in reports for key in ("source_validation", "launch_validation")):
                raise AssertionError("rank validation strategies differ")
            summary = {
                "schema_version": 2,
                "mode": args.mode,
                "passed": all(r["passed"] for r in reports),
                "device_executed": all(r["device_executed"] for r in reports),
                "iterations": args.iterations,
                "warmup": args.warmup,
                "correctness_scope": "benchmark_workload_only",
                "full_device_acceptance": "not_assessed",
                "execution_policy": result["execution_policy"],
                "source_validation": result["source_validation"],
                "launch_validation": result["launch_validation"],
                "workload": result["workload"],
                "ranks": reports,
                "local_compute_blocks": 1 if args.mode == "p0b" else 2 * mt * nt,
                "local_compute_launches": 0 if args.mode == "p0b" else 1,
                "timing_basis": "host_wall; packing, staging, preparation, readback and unpacking in E2E; dense FP64 oracle excluded",
                "stage_aggregation": "max of two ranks per iteration then statistics; stage maxima and medians are not additive",
                **paired_summary(reports, args.iterations),
            }
            intervals = [[sample for sample in report["timing_samples"] if not sample["warmup"]] for report in reports]
            interval_keys = set(intervals[0][0]["device_intervals_seconds"])
            if any(set(sample["device_intervals_seconds"]) != interval_keys for samples in intervals for sample in samples):
                raise AssertionError("device timing coverage differs across iterations or ranks")
            summary["device_intervals"] = {
                key: describe_samples(
                    max(pair[0]["device_intervals_seconds"][key], pair[1]["device_intervals_seconds"][key]) for pair in zip(*intervals)
                )
                for key in sorted(interval_keys)
            }
            summary["device_timing_basis"] = (
                "event_stream_interval_including_submission_gaps; separate P0-A streams are not summed" if args.measure_device else None
            )
            write_json(root / "detail.json", summary)
    except BaseException as error:
        result["exit_code"] = 1
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        traceback.print_exc()
        raise
    finally:
        result.setdefault("exit_code", 0)
        write_json(root / f"rank{rank}.json", result)
    return 0


def milliseconds(value):
    return value * 1000 if type(value) in (int, float) and 0 <= value <= 1e100 and math.isfinite(value) else None


def shape_label(shape):
    return "×".join(map(str, shape)) if shape is not None else "-"


def compact_case(entry):
    """Fixed-size, allowlisted export; detailed evidence stays in each case."""
    summary = entry.get("summary", {})
    ranks = summary.get("ranks", [])
    result = {key: entry[key] for key in ("shape", "mode", "stage", "status", "exit_code", "timed_out", "device_executed")}
    result.update(
        execution_policy=summary.get("execution_policy"),
        source_validation=summary.get("source_validation"),
        launch_validation=summary.get("launch_validation"),
    )
    result["device_intervals_median_ms"] = {
        key: milliseconds(summary.get("device_intervals", {}).get(key, {}).get("median_seconds"))
        for key in ("collective_and_local_gemm", "collective", "communication", "local_gemm", "fused")
    }
    checks = [check for rank in ranks for check in rank.get("checks", [])]
    result["max_abs_error"] = max(
        (check["max_abs_error"] for check in checks if isinstance(check.get("max_abs_error"), (int, float))), default=None
    )
    for boundary in ("execution", "end_to_end"):
        stats = summary.get(boundary, {})
        result[boundary] = {
            "count": stats.get("count"),
            "median_ms": milliseconds(stats.get("median_seconds")),
            "p95_ms": milliseconds(stats.get("p95_seconds")),
        }
        elapsed = stats.get("median_seconds")
        work = workload(*entry["shape"]) if entry["shape"] is not None else {}
        result[boundary]["useful_tflops_per_rank"] = work["flops_per_rank"] / elapsed / 1e12 if elapsed and elapsed > 0 else None
        result[boundary]["logical_peer_gbytes_per_second"] = (
            work["logical_peer_payload_bytes_per_rank"] / elapsed / 1e9 if elapsed and elapsed > 0 else None
        )
    for field in ("first_build_or_cache_load_seconds", "first_execution_seconds", "first_end_to_end_seconds"):
        values = [rank.get(field) for rank in ranks]
        result[field.removesuffix("_seconds") + "_ms"] = (
            max(values) * 1e3 if len(values) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in values) else None
        )
    for field, allowed in (
        (
            "stages",
            (
                "pack",
                "prepare",
                "staging",
                "staging_sync",
                "collective_submit",
                "launch",
                "complete",
                "local_gemm_submit",
                "final_sync",
                "readback",
                "unpack",
            ),
        ),
        (
            "native_run",
            (
                "validate_bindings",
                "plan_validation",
                "plan_preflight",
                "build_or_cache_validation",
                "build_preflight",
                "bind_and_submit",
                "stream_sync",
                "report_finalize",
                "total",
            ),
        ),
    ):
        result[field + "_median_ms"] = {key: milliseconds(summary.get(field, {}).get(key, {}).get("median_seconds")) for key in allowed}
    result["error_types"] = sorted(
        {
            rank.get("error_type")
            for rank in entry.get("rank_reports", [])
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", rank.get("error_type", ""))
        }
    )
    result["rank_exit_codes"] = [
        next((rank.get("exit_code") for rank in entry.get("rank_reports", []) if rank.get("rank") == index), None) for index in range(2)
    ]
    result["compile_build_ms"] = milliseconds(summary.get("first_build_or_cache_load_seconds"))
    return result


def save_sweep(root, entries, commit, environment=None):
    compact = [compact_case(entry) for entry in entries]
    for entry in compact:
        baseline = next(
            (
                other
                for other in compact
                if other["shape"] == entry["shape"]
                and other["stage"] == entry["stage"]
                and other["mode"] == "hccl"
                and other["status"] == "passed"
            ),
            None,
        )
        for boundary in ("execution", "end_to_end"):
            own, base = entry[boundary]["median_ms"], baseline[boundary]["median_ms"] if baseline else None
            entry[boundary]["hccl_over_candidate"] = base / own if base and own and entry["status"] == "passed" else None
            entry[boundary]["candidate_over_hccl"] = own / base if base and own and entry["status"] == "passed" else None
    report = {
        "schema_version": 1,
        "report_kind": "p0_dense_shape_sweep",
        "commit": commit,
        "environment": environment,
        "passed": bool(entries) and all(entry["status"] == "passed" for entry in entries),
        "correctness_scope": "benchmark_workload_only",
        "full_device_acceptance": "not_assessed",
        "cases": compact,
    }
    write_json(root / "summary.json", report)
    lines = [
        "# P0 dense AllGather + GEMM shape sweep",
        "",
        f"Commit: {commit or 'unknown'}",
        f"Environment: {json.dumps(environment, sort_keys=True)}",
        "Scope: only these measured workloads; full distributed acceptance is not assessed.",
        "M is per rank; output is (2M,N). K is a real accumulation dimension. Each peer sends one whole M×K BF16 packet.",
        "HCCL/P0-A: one grid GEMM (2×M/64×N/64 blocks); P0-B: one mixed block. Core counts differ.",
        "Execution is synchronized host wall time. E2E includes pack/staging/prepare/readback/unpack; oracle is outside E2E.",
        "Throughput is useful FLOPs/logical peer bytes divided by whole-workload time; it is not physical network bandwidth.",
        "",
        "| Stage | M×N×K | Mode | Status / exit / timeout | N | Exec med/p95 ms | E2E med/p95 ms | HCCL/exec | Exec TFLOP/s | Max error |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "null" if value is None else f"{value:.6g}"

    for case in compact:
        exe, e2e = case["execution"], case["end_to_end"]
        lines.append(
            f"| {case['stage']} | {shape_label(case['shape'])} | {case['mode']} | {case['status']} / {case['exit_code']} / {case['timed_out']} | {exe['count']} | {number(exe['median_ms'])}/{number(exe['p95_ms'])} | {number(e2e['median_ms'])}/{number(e2e['p95_ms'])} | {number(exe['hccl_over_candidate'])} | {number(exe['useful_tflops_per_rank'])} | {number(case['max_abs_error'])} |"
        )
    device_cases = [case for case in compact if case["stage"] == "device"]
    if device_cases:
        lines += [
            "",
            "Event intervals include stream idle gaps; P0-A intervals are on separate streams and must not be summed.",
            "| M×N×K / mode | Policy / source / launch guard | First build / exec / E2E ms | HCCL+GEMM event ms | HCCL event ms | Comm event ms | Local GEMM event ms | Fused event ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for case in device_cases:
            intervals = case["device_intervals_median_ms"]
            first = "/".join(number(case[key + "_ms"]) for key in ("first_build_or_cache_load", "first_execution", "first_end_to_end"))
            lines.append(
                f"| {'×'.join(map(str, case['shape']))} / {case['mode']} | {case['execution_policy']} / {case['source_validation']} / {case['launch_validation']} | {first} | {number(intervals['collective_and_local_gemm'])} | {number(intervals['collective'])} | {number(intervals['communication'])} | {number(intervals['local_gemm'])} | {number(intervals['fused'])} |"
            )
        lines += [
            "",
            "Host stage medians are paired rank maxima, not additive. Native timings are nested inside launch.",
            "| M×N×K / mode | Pack | Prepare | Launch | Complete | Readback+unpack* | Native plan | Native build/cache | Native submit | Native sync |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for case in device_cases:
            outer, native = case["stages_median_ms"], case["native_run_median_ms"]
            values = [number(outer[key]) for key in ("pack", "prepare", "launch", "complete")]
            values += [f"{number(outer['readback'])} / {number(outer['unpack'])}"]
            values += [number(native[key]) for key in ("plan_validation", "build_or_cache_validation", "bind_and_submit", "stream_sync")]
            lines.append(f"| {'×'.join(map(str, case['shape']))} / {case['mode']} | " + " | ".join(values) + " |")
        lines.append("* Readback / unpack are separate medians; the slash does not denote their sum. All stage values are ms.")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="compile", help="Comma-separated cpu,compile,device; no device execution by default")
    parser.add_argument("--shapes", default="128,256,1024", help="Sizes or MxNxK, each dimension a multiple of 64 up to 1024")
    parser.add_argument("--modes", default="hccl,p0a,p0b")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--execution-policy", choices=("artifact", "strict"), default="artifact")
    parser.add_argument(
        "--measure-device", action="store_true", help="Also measure actual stream event intervals, including submission gaps"
    )
    parser.add_argument("--sdk-profile", choices=("native", "cann-9.1"), default="cann-9.1")
    parser.add_argument("--shmem-root", default=os.environ.get("SHMEM_ROOT"))
    parser.add_argument("--shmem-library", default=os.environ.get("SHMEM_LIBRARY"))
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--operation-timeout", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("output/distributed/shapes"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("compile", "device"), default="compile", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("hccl", "p0a", "p0b"), default="p0b", help=argparse.SUPPRESS)
    for dimension in ("m", "n", "k"):
        parser.add_argument("--matrix-" + dimension, dest=dimension, type=int, default=1024, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-port", type=int, default=29720, help=argparse.SUPPRESS)
    parser.add_argument("--control-port", type=int, default=29721, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        shapes = parse_shapes(args.shapes)
        tile_counts(args.m, args.n, args.k)
    except ValueError as error:
        parser.error(str(error))
    stages, modes = args.stages.split(","), args.modes.split(",")
    if not stages or len(stages) != len(set(stages)) or any(stage not in ("cpu", "compile", "device") for stage in stages):
        parser.error("stages must be unique cpu,compile,device")
    if not modes or len(modes) != len(set(modes)) or any(mode not in ("hccl", "p0a", "p0b") for mode in modes):
        parser.error("modes must be unique hccl,p0a,p0b")
    if (
        args.iterations < 1
        or args.warmup < 0
        or not math.isfinite(args.timeout)
        or not math.isfinite(args.operation_timeout)
        or min(args.timeout, args.operation_timeout) <= 0
    ):
        parser.error("iterations/timeouts must be positive and warmup nonnegative")
    if min(args.devices) < 0 or len(set(args.devices)) != 2:
        parser.error("devices must be distinct and nonnegative")
    if args.worker:
        return worker(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run_", dir=args.output_dir.resolve()))
    provenance = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    commit = (
        provenance.stdout.strip()
        if provenance.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", provenance.stdout.strip())
        else None
    )
    versions = {}
    for package in ("tilelang", "torch", "torch_npu", "apache-tvm-ffi", "numpy", "pytest"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        versions[package] = version if version and re.fullmatch(r"[A-Za-z0-9.+_-]{1,128}", version) else None
    shmem_revision = None
    if args.shmem_root:
        revision = subprocess.run(["git", "-C", args.shmem_root, "rev-parse", "HEAD"], capture_output=True, text=True)
        if revision.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision.stdout.strip()):
            shmem_revision = revision.stdout.strip()
    environment = {"shmem_revision": shmem_revision, "sdk_profile": args.sdk_profile, "versions": versions}
    entries = []
    for stage in stages:
        if stage == "cpu":
            process = run_command(
                [sys.executable, "-m", "pytest", "-q", "examples/distributed/test_p0_shapes.py"], root / "cpu", args.timeout
            )
            entries.append(
                {
                    "shape": None,
                    "mode": "cpu",
                    "stage": "cpu",
                    "status": "passed" if process["exit_code"] == 0 and not process["timed_out"] else "failed",
                    "exit_code": process["exit_code"],
                    "timed_out": process["timed_out"],
                    "device_executed": False,
                }
            )
            continue
        for shape in shapes:
            for mode in modes:
                directory = root / stage / ("x".join(map(str, shape))) / mode
                with socket.socket() as first, socket.socket() as second:
                    first.bind(("127.0.0.1", 0))
                    second.bind(("127.0.0.1", 0))
                    ports = [first.getsockname()[1], second.getsockname()[1]]
                command = [sys.executable]
                if stage == "device":
                    command += [
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=2",
                        "--log-dir",
                        str(directory / "logs"),
                        "--redirects",
                        "3",
                    ]
                command += [
                    "-m",
                    "examples.distributed.benchmark_p0_shapes",
                    "--worker",
                    "--stage",
                    stage,
                    "--mode",
                    mode,
                    "--matrix-m",
                    str(shape[0]),
                    "--matrix-n",
                    str(shape[1]),
                    "--matrix-k",
                    str(shape[2]),
                    "--iterations",
                    str(args.iterations),
                    "--warmup",
                    str(args.warmup),
                    "--execution-policy",
                    args.execution_policy,
                    "--sdk-profile",
                    args.sdk_profile,
                    "--operation-timeout",
                    str(args.operation_timeout),
                    "--bootstrap-port",
                    str(ports[0]),
                    "--control-port",
                    str(ports[1]),
                    "--devices",
                    *map(str, args.devices),
                    "--output-dir",
                    str(directory),
                ]
                if args.shmem_root:
                    command += ["--shmem-root", args.shmem_root]
                if args.shmem_library:
                    command += ["--shmem-library", args.shmem_library]
                if args.measure_device:
                    command += ["--measure-device"]
                process = run_command(
                    command, directory, args.timeout, env_overrides={"P0_SHAPES_CACHE_ROOT": str(directory / "cold-cache")}
                )
                evidence = directory / ("build.json" if stage == "compile" else "detail.json")
                summary = json.loads(evidence.read_text()) if evidence.is_file() else {}
                if stage == "device":
                    assessment = assess_performance(process, summary, iterations=args.iterations)
                else:
                    assessment = {
                        "status": "passed"
                        if process["exit_code"] == 0
                        and not process["timed_out"]
                        and summary.get("compiled")
                        and summary.get("device_executed") is False
                        else "failed",
                        "device_executed": False,
                    }
                entry = {
                    "shape": list(shape),
                    "mode": mode,
                    "stage": stage,
                    **assessment,
                    "exit_code": process["exit_code"],
                    "timed_out": process["timed_out"],
                    "summary": summary,
                    "rank_reports": [
                        json.loads(path.read_text()) for rank in range(2) if (path := directory / f"rank{rank}.json").is_file()
                    ],
                }
                entries.append(entry)
                write_json(root / "detail.json", {"cases": entries})
                save_sweep(root, entries, commit, environment)
                print(f"{stage} {'x'.join(map(str, shape))} {mode}: {entry['status']}", flush=True)
    report = save_sweep(root, entries, commit, environment)
    print(f"Results: {root}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
