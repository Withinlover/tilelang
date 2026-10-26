"""Call two distributed JIT specializations in one fixed-heap Session.

Launch two workers from the repository root after configuring CANN and SHMEM:
    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        --module examples.distributed.example_jit --sdk-profile cann-9.1

The launcher supplies RANK and LOCAL_RANK. No torch process group is created.
Each worker checks only its own output against an independent FP64 oracle.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import tilelang.distributed as D

from examples.distributed.example_allgather_gemm import allgather_gemm, make_inputs


def check_output(actual, original_a, original_b):
    """Reconstruct expected output from original BF16 inputs on the host."""
    a = (original_a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b = (original_b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    expected = np.swapaxes(a @ b, 0, 1)
    if actual.dtype != np.float32 or actual.shape != expected.shape:
        raise AssertionError(f"expected float32 output of shape {expected.shape}, got {actual.dtype} {actual.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    return {"shape": list(actual.shape), "max_abs_error": float(np.abs(actual.astype(np.float64) - expected).max())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-profile", choices=("native", "cann-9.1"), default="cann-9.1")
    parser.add_argument("--shmem-root", default=os.environ.get("SHMEM_ROOT"))
    parser.add_argument("--shmem-library", default=os.environ.get("SHMEM_LIBRARY"))
    parser.add_argument("--bootstrap-port", type=int, default=29610)
    parser.add_argument("--control-port", type=int, default=29611)
    parser.add_argument("--heap-mib", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, help="optional per-rank compiler and launch diagnostics")
    args = parser.parse_args(argv)
    if os.environ.get("WORLD_SIZE") != "2" or os.environ.get("RANK") not in ("0", "1"):
        parser.error("launch exactly two workers with torch.distributed.run")
    if not args.shmem_root:
        parser.error("set SHMEM_ROOT or pass --shmem-root pointing to the pinned SHMEM source checkout")
    if args.heap_mib <= 0:
        parser.error("--heap-mib must be positive")
    if not all(1 <= port <= 65535 for port in (args.bootstrap_port, args.control_port)):
        parser.error("ports must be in 1..65535")
    if args.bootstrap_port == args.control_port:
        parser.error("SHMEM bootstrap and metadata control require different ports")

    rank = int(os.environ["RANK"])
    device = int(os.environ.get("LOCAL_RANK", str(rank)))
    compile_options = {
        "compile_profile": "cann-9.1-gemm64" if args.sdk_profile == "cann-9.1" else "native",
        "transport_compile_profile": "cann-9.1-shmem" if args.sdk_profile == "cann-9.1" else "native",
    }
    if args.output_dir is not None:
        compile_options.update(output_dir=args.output_dir / f"rank{rank}", debug=True)

    @D.jit(inputs=("Send", "B"), outputs=("Output",), epochs="rounds", profile=D.NATIVE_PROFILE, **compile_options)
    def gemm(rounds=3, slots=2):
        return allgather_gemm(rounds=rounds, slots=slots)

    checks = []
    with D.init(
        profile=D.NATIVE_PROFILE,
        rank=rank,
        device=device,
        bootstrap=f"tcp://127.0.0.1:{args.bootstrap_port}",
        control_port=args.control_port,
        shmem_root=args.shmem_root,
        shmem_library=args.shmem_library,
        heap_bytes=args.heap_mib * 1024 * 1024,
    ) as session:
        # Every rank binds and invokes these specializations in the same order.
        # A and B each retain their own workspace inside this Session's heap.
        op_a = gemm(rounds=3, slots=2).bind(session)
        op_b = gemm(rounds=1, slots=2).bind(session)
        first_output = saved_output = None
        for label, op, rounds, seed in (("A", op_a, 3, 0), ("B", op_b, 1, 1), ("A", op_a, 3, 2)):
            # Both workers generate the same immutable global inputs for the
            # oracle; only this rank's operand is copied into its Send buffer.
            original_a, original_b = make_inputs(rounds, seed)
            actual = op(original_a[rank], original_b)
            checks.append({"operator": label, "seed": seed, **check_output(actual, original_a, original_b)})
            if first_output is None:
                first_output, saved_output = actual, actual.copy()
            else:
                np.testing.assert_array_equal(first_output, saved_output)

    result = {
        "rank": rank,
        "device": device,
        "local_checks_passed": True,
        "checks": checks,
        "first_output_preserved": True,
        "session_closed": session.state == "CLOSED",
    }
    if args.output_dir is not None:
        rank_dir = args.output_dir / f"rank{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        (rank_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
