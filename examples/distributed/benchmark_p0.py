"""Two-worker timing of the same BF16 AllGather + local 64x64 GEMM workload.

Launch using run_p0_validation.py --stages performance. Execution timings are
synchronized host wall times, including submission and protocol bookkeeping;
they are not isolated device-event latency or a claim of communication overlap.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time
import traceback

if os.environ.get("P0_BENCH_CACHE_ROOT"):
    os.environ["TILELANG_CACHE_DIR"] = str(Path(os.environ["P0_BENCH_CACHE_ROOT"]) / f"rank{os.environ.get('RANK', 'unknown')}")

import tilelang
import tilelang.ascend.language as T
import tilelang.distributed as D

from examples.distributed.example_allgather_gemm import TILE, allgather_gemm, make_inputs
from examples.distributed.example_jit import check_output
from examples.distributed.run_native_distributed import exchange


def local_gemm():
    @T.prim_func
    def program(A: T.Tensor((TILE, TILE), "bfloat16"), B: T.Tensor((TILE, TILE), "bfloat16"), C: T.Tensor((TILE, TILE), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            a_l0 = T.alloc_l0a((TILE, TILE), "bfloat16")
            b_l0 = T.alloc_l0b((TILE, TILE), "bfloat16")
            c_l0 = T.alloc_l0c((TILE, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0, transpose=True)
            T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=True)
            T.copy(c_l0, C)

    return program


def describe_samples(values):
    return {"samples_seconds": values, "median_seconds": statistics.median(values), "min_seconds": min(values), "max_seconds": max(values)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hccl", "p0a", "p0b"), required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--sdk-profile", choices=("native", "cann-9.1"), default="cann-9.1")
    parser.add_argument("--shmem-root", default=os.environ.get("SHMEM_ROOT"))
    parser.add_argument("--shmem-library", default=os.environ.get("SHMEM_LIBRARY"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--bootstrap-port", type=int, default=29710)
    parser.add_argument("--control-port", type=int, default=29711)
    args = parser.parse_args(argv)
    if os.environ.get("WORLD_SIZE") != "2" or os.environ.get("RANK") not in ("0", "1"):
        parser.error("requires exactly two torchrun workers")
    if min(args.iterations, args.rounds, args.slots) <= 0 or args.warmup < 0:
        parser.error("iterations/rounds/slots must be positive; warmup must be nonnegative")
    if len(set(args.devices)) != 2 or min(args.devices) < 0:
        parser.error("select two distinct nonnegative device IDs")
    if not all(1 <= port <= 65535 for port in (args.bootstrap_port, args.control_port)) or args.bootstrap_port == args.control_port:
        parser.error("bootstrap/control ports must be distinct and in 1..65535")
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    rank = int(os.environ["RANK"])
    device = args.devices[rank]
    torch.npu.set_device(device)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = {"rank": rank, "mode": args.mode, "passed": False, "device_executed": False, "checks": []}
    session = None
    try:
        started = time.perf_counter()
        configs = {"tl.ascend_sdk_compatibility": "cann-9.1-gemm64"} if args.sdk_profile == "cann-9.1" else {}
        gemm = (
            tilelang.compile(local_gemm(), target={"kind": "ascend", "arch": "dav-3510"}, execution_backend="tvm_ffi", pass_configs=configs)
            if args.mode != "p0b"
            else None
        )
        result["first_build_or_cache_load_seconds"] = time.perf_counter() - started
        if args.mode == "hccl":
            tensor_a = torch.empty((args.rounds, TILE, TILE), dtype=torch.bfloat16, device=f"npu:{device}")
            gathered = [torch.empty_like(tensor_a) for _ in range(2)]
        else:
            profile = "ascend950_shmem_mte_p0a_v1" if args.mode == "p0a" else "ascend950_shmem_mte_p0b_v1"
            source = exchange(args.rounds, args.slots) if args.mode == "p0a" else allgather_gemm(args.rounds, args.slots)
            compiled = D.compile(
                source,
                profile=profile,
                mode="native",
                shmem_root=args.shmem_root,
                compile_profile="cann-9.1-gemm64" if args.sdk_profile == "cann-9.1" else "native",
                transport_compile_profile="cann-9.1-shmem" if args.sdk_profile == "cann-9.1" else "native",
                debug=False,
            )
            compiled.build(shmem_root=args.shmem_root, epochs=args.rounds)
        if args.mode != "hccl":
            result["first_build_or_cache_load_seconds"] = time.perf_counter() - started
        if args.mode == "hccl":
            dist.init_process_group("hccl", timeout=timedelta(seconds=120))
        else:
            session = D.init(
                profile=profile,
                rank=rank,
                device=device,
                bootstrap=f"tcp://127.0.0.1:{args.bootstrap_port}",
                control_port=args.control_port,
                shmem_root=args.shmem_root,
                shmem_library=args.shmem_library,
                timeout=120,
            )
            objects = {}
            for spec in compiled.manifest["objects"]:
                common = dict(key=spec["key"], shape=spec["shape"], dtype=spec["dtype"])
                objects[spec["key"]] = (
                    D.alloc_signals(session, **common)
                    if spec["kind"] == "signal"
                    else D.alloc_symmetric(session, remote_access=spec["remote_access"], **common)
                )
            channel = D.channel(session, key="comm")
        tensor_b = torch.empty((TILE, TILE), dtype=torch.bfloat16, device=f"npu:{device}") if gemm is not None else None
        tensor_out = torch.empty((args.rounds, 2, TILE, TILE), dtype=torch.float32, device=f"npu:{device}") if gemm is not None else None
        execution_samples, end_to_end_samples = [], []
        for iteration in range(args.warmup + args.iterations):
            a, b = make_inputs(args.rounds, 1000 + iteration)
            if args.mode == "hccl":
                dist.barrier()
            else:
                session.preflight({"benchmark_iteration": iteration})
            torch.npu.synchronize(device)
            e2e_started = time.perf_counter()
            if args.mode == "hccl":
                tensor_a.copy_(torch.from_numpy(a[rank].copy()).view(torch.bfloat16))
            else:
                ticket = D.prepare_launch(
                    session, compiled, channel=channel, epochs=args.rounds, protocol="p0-benchmark-v1", parameters={"iteration": iteration}
                )
                D.copy_in(objects["Input" if args.mode == "p0a" else "Send"], a[rank], ticket=ticket)
                if args.mode == "p0b":
                    D.copy_in(objects["B"], b, ticket=ticket)
            if tensor_b is not None:
                tensor_b.copy_(torch.from_numpy(b.copy()).view(torch.bfloat16))
            torch.npu.synchronize(device)
            execution_started = time.perf_counter()
            if args.mode == "hccl":
                dist.all_gather(gathered, tensor_a)
                for row in range(args.rounds):
                    for owner in range(2):
                        gemm(gathered[owner][row], tensor_b, tensor_out[row, owner])
            else:
                execution = D.complete(D.launch(compiled, ticket=ticket))
                if execution.get("physical_kernel_count") != 1:
                    raise AssertionError("fused benchmark requires exactly one communication/fused device kernel")
                if args.mode == "p0a":
                    # Borrow the existing symmetric GM data directly. D.complete
                    # has synchronized its stream; no host bounce of payloads.
                    for row in range(args.rounds):
                        for owner in range(2):
                            buffer = objects["Input" if owner == rank else "Output"]
                            view = buffer.view(shape=(TILE, TILE), byte_offset=row * TILE * TILE * 2)
                            gemm(view, tensor_b, tensor_out[row, owner])
                            del view
            torch.npu.synchronize(device)
            elapsed = time.perf_counter() - execution_started
            actual = tensor_out.cpu().numpy() if tensor_out is not None else session.copy_out(objects["Output"])
            e2e_elapsed = time.perf_counter() - e2e_started
            result["device_executed"] = True
            result["checks"].append({"iteration": iteration, **check_output(actual, a, b)})
            if iteration == 0:
                result["first_execution_seconds"] = elapsed
                result["first_end_to_end_seconds"] = e2e_elapsed
            if iteration >= args.warmup:
                execution_samples.append(elapsed)
                end_to_end_samples.append(e2e_elapsed)
        result.update(passed=True, execution=describe_samples(execution_samples), end_to_end=describe_samples(end_to_end_samples))
        reports = [None, None]
        if session is not None:
            reports = session.control.all_gather(result)
            D.close(session)
        else:
            dist.all_gather_object(reports, result)
            dist.destroy_process_group()
        if rank == 0:
            # Ranks run concurrently: use the slower rank for each paired sample.
            summary = {
                "schema_version": 1,
                "mode": args.mode,
                "passed": all(item["passed"] for item in reports),
                "device_executed": all(item["device_executed"] for item in reports),
                "rounds": args.rounds,
                "slots": args.slots,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "tile": TILE,
                "dtype": "bfloat16",
                "output_dtype": "float32",
                "first_build_or_cache_load_seconds": [item["first_build_or_cache_load_seconds"] for item in reports],
                "first_execution_seconds": [item["first_execution_seconds"] for item in reports],
                "first_end_to_end_seconds": [item["first_end_to_end_seconds"] for item in reports],
                "timing_basis": "synchronized host wall time; execution includes host submission and protocol preflight; e2e additionally includes allocation-free input staging, ticket preparation, output readback; oracle excluded",
                "p0a_compute": "one pure AIV exchange plus ordinary local TileLang Cube kernels",
            }
            for boundary in ("execution", "end_to_end"):
                samples = [max(pair) for pair in zip(*(item[boundary]["samples_seconds"] for item in reports))]
                summary[boundary] = describe_samples(samples)
            (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        raise
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
