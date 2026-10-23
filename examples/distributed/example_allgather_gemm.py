"""Two-rank AllGather + GEMM expressed once in the P0 source DSL.

Run with CANNSIM_ROOT pointing to the extended cannsim-helper checkout:
    python examples/distributed/example_allgather_gemm.py --mode protocol
    python examples/distributed/example_allgather_gemm.py --mode rank-segments

Communication executes in the CPU protocol model. In rank-segments mode each
local GEMM is extracted from this PrimFunc, compiled for Ascend 950, and run in
a fresh single-device CANN simulation. No physical multi-device timing is
inferred from the logical protocol or independent computation traces.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import tilelang.ascend.language as T
import tilelang.distributed as D
import tilelang.language.distributed as TD


TILE = 64
WORLD_SIZE = 2


def allgather_gemm(rounds=3, slots=2):
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= 10000:
        raise ValueError("rounds must be an integer in 1..10000")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
        raise ValueError("slots must be a positive integer")
    group = TD.GroupSpec("world", WORLD_SIZE)
    channel = TD.ChannelSpec("comm", group)

    @T.prim_func
    def program(
        Send: TD.SymBuffer((rounds, TILE, TILE), "bfloat16", group=group, remote_access="read"),
        B: TD.SymBuffer((TILE, TILE), "bfloat16", group=group, remote_access="read"),
        Recv: TD.SymBuffer((slots, TILE, TILE), "bfloat16", group=group, remote_access="write"),
        Ready: TD.SignalBuffer((slots,), group=group),
        Ack: TD.SignalBuffer((slots,), group=group),
        Output: TD.SymBuffer((rounds, WORLD_SIZE, TILE, TILE), "float32", group=group, remote_access="write"),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            a_l0 = T.alloc_l0a((TILE, TILE), "bfloat16")
            b_l0 = T.alloc_l0b((TILE, TILE), "bfloat16")
            c_l0 = T.alloc_l0c((TILE, TILE), "float32")
            rank = TD.rank(group)
            peer = 1 - rank
            for iteration in T.serial(rounds):
                slot = iteration % slots
                epoch = epoch0 + iteration + 1
                if iteration >= slots:
                    TD.wait_signal(Ack[slot], epoch - slots, channel=channel, covers=[])
                sent = TD.put_signal(
                    Send[iteration, :, :],
                    TD.remote(Recv, peer)[slot, :, :],
                    signal=TD.remote(Ready, peer)[slot],
                    value=epoch,
                    channel=channel,
                )
                TD.wait(sent, completion="local")
                TD.wait_signal(Ready[slot], epoch, channel=channel, covers=[Recv[slot, :, :]])

                T.copy(Send[iteration, :, :], a_l1)
                T.copy(B, b_l1)
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0, transpose=True)
                T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=True)
                T.copy(c_l0, Output[iteration, rank, :, :])

                T.copy(Recv[slot, :, :], a_l1)
                T.copy(B, b_l1)
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0, transpose=True)
                T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=True)
                T.copy(c_l0, Output[iteration, peer, :, :])

                TD.signal(TD.remote(Ack, peer)[slot], epoch, channel=channel, release_regions=[Recv[slot, :, :]])
            for iteration in T.serial(max(0, rounds - slots), rounds):
                TD.wait_signal(Ack[iteration % slots], epoch0 + iteration + 1, channel=channel, covers=[])
            TD.quiet(channel=channel)

    return program


def default_helper_root():
    configured = os.environ.get("CANNSIM_ROOT")
    return Path(configured).expanduser().resolve() if configured else None


def make_inputs(rounds, seed):
    rng = np.random.default_rng(seed)
    values = rng.uniform(-0.5, 0.5, (WORLD_SIZE, rounds, TILE, TILE)).astype("<f4")
    values += np.arange(WORLD_SIZE, dtype=np.float32)[:, None, None, None] * 0.125
    values += np.arange(rounds, dtype=np.float32)[None, :, None, None] * 0.03125
    a = (values.view("<u4") >> 16).astype("<u2")
    b = (rng.uniform(-0.5, 0.5, (TILE, TILE)).astype("<f4").view("<u4") >> 16).astype("<u2")
    a.flags.writeable = b.flags.writeable = False
    return a, b


def global_accuracy(actual, original_a, original_b, *, rtol=1e-5, atol=1e-5):
    """FP64 oracle uses immutable original inputs, never received payloads."""
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    a = (original_a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b = (original_b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    expected = np.swapaxes(a @ b, 0, 1)
    expected = np.broadcast_to(expected, (WORLD_SIZE, *expected.shape))
    if actual.shape != expected.shape:
        raise ValueError(f"global output shape {actual.shape} differs from expected {expected.shape}")
    finite = np.isfinite(actual)
    error = np.abs(actual.astype(np.float64) - expected)
    matching = finite & (error <= atol + rtol * np.abs(expected))
    return {
        "passed": bool(matching.all()),
        "reference": "FP64 matmul of original global BF16 inputs",
        "shape": list(actual.shape),
        "elements": int(actual.size),
        "mismatches": int((~matching).sum()),
        "nonfinite": int((~finite).sum()),
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": float(error[finite].max()) if finite.any() else None,
        "ranks": [
            {"rank": rank, "passed": bool(matching[rank].all()), "mismatches": int((~matching[rank]).sum())} for rank in range(WORLD_SIZE)
        ],
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(
    *,
    mode="protocol",
    rounds=3,
    slots=2,
    seed=0,
    fault=None,
    helper_root=None,
    output_dir=None,
    consumer_delay=7,
    segment_timeout=120,
    rtol=1e-5,
    atol=1e-5,
):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    program = allgather_gemm(rounds, slots)
    kernel = D.compile(program, mode=mode, helper_root=helper_root or default_helper_root(), output_dir=output_dir)
    session = D.init()
    allocations = {}
    for spec in kernel.manifest["objects"]:
        common = dict(key=spec["key"], shape=spec["shape"], dtype=spec["dtype"])
        allocations[spec["key"]] = (
            D.alloc_signals(session, **common)
            if spec["kind"] == "signal"
            else D.alloc_symmetric(session, remote_access=spec["remote_access"], **common)
        )
    channel = D.channel(session, key="comm")
    ticket = D.prepare_launch(
        session,
        kernel,
        channel=channel,
        epochs=rounds,
        protocol="allgather-gemm-v1",
        parameters={"rounds": rounds, "slots": slots, "tile": TILE},
    )
    original_a, original_b = make_inputs(rounds, seed)
    D.copy_in(allocations["Send"], {rank: original_a[rank] for rank in range(WORLD_SIZE)}, ticket=ticket)
    D.copy_in(allocations["B"], {rank: original_b for rank in range(WORLD_SIZE)}, ticket=ticket)
    D.copy_in(
        allocations["Output"],
        {rank: np.full((rounds, WORLD_SIZE, TILE, TILE), np.nan, dtype="<f4") for rank in range(WORLD_SIZE)},
        ticket=ticket,
    )
    try:
        launch = D.launch(kernel, ticket=ticket, seed=seed, fault=fault, consumer_delay=consumer_delay, segment_timeout=segment_timeout)
        adapter_report = D.complete(launch)
    except D.RunFailedError as error:
        adapter_report = error.run.result
    directory = Path(adapter_report["run_dir"])
    _write_json(directory / "adapter_manifest.json", adapter_report)
    np.savez(directory / "oracle_inputs.npz", original_a=original_a, original_b=original_b)
    actual = np.stack([allocations["Output"].arrays[rank] for rank in range(WORLD_SIZE)])
    np.save(directory / "outputs.npy", actual)
    accuracy = global_accuracy(actual, original_a, original_b, rtol=rtol, atol=atol)
    _write_json(directory / "accuracy.json", accuracy)
    expected_segments = WORLD_SIZE * WORLD_SIZE * rounds if mode == "rank-segments" else 0
    segments = adapter_report["segments"]
    segment_complete = len(segments) == expected_segments and all(s.get("passed") and s.get("report_complete") for s in segments)
    protocol = adapter_report["protocol"]
    fault_observed = fault is None or bool(protocol.get("fault_injected"))
    report = {
        **adapter_report,
        "adapter_passed": adapter_report["passed"],
        "global_accuracy": accuracy,
        "rounds": rounds,
        "slots": slots,
        "segments_expected": expected_segments,
        "segments_completed": sum(bool(s.get("passed")) for s in segments),
        "report_complete": segment_complete,
        "oracle_inputs_sha256": hashlib.sha256((directory / "oracle_inputs.npz").read_bytes()).hexdigest(),
        "limitations": [
            "Communication and inter-rank ordering are a CPU semantic model.",
            "CANN compute segments run independently on device 0; no peer DMA is executed.",
            "Physical multi-device cache coherence, performance and comm_scheduled lowering are not certified.",
        ],
    }
    report["passed"] = bool(
        adapter_report["passed"] and protocol.get("passed") and accuracy["passed"] and segment_complete and fault_observed
    )
    if not accuracy["passed"]:
        report["errors"] = [*report["errors"], f"global accuracy: {accuracy['mismatches']} mismatches"]
    if not fault_observed:
        report["errors"] = [*report["errors"], "requested fault was not injected"]
    _write_json(directory / "manifest.json", report)
    (directory / "report.md").write_text(
        f"# P0 DSL AllGather + GEMM: {'PASS' if report['passed'] else 'FAIL'}\n\n"
        f"Mode `{mode}`; two logical ranks, {rounds} rounds, {slots} slots. "
        f"Global mismatches: {accuracy['mismatches']}/{accuracy['elements']}; "
        f"maximum absolute error: {accuracy['max_abs_error']}.\n\n"
        "Communication and computation originate from [one PrimFunc](program.tir). "
        "[Serialized IR](program.json), [ABI](abi.json), [global accuracy](accuracy.json), "
        "[protocol trace](protocol_trace.json), [segment trace index](trace_index.json), "
        "and [manifest](manifest.json) preserve the validation evidence.\n\n"
        + "\n".join(f"- {item}" for item in report["limitations"])
        + "\n"
    )
    if session.state != "FAILED":
        D.close(session)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("protocol", "rank-segments"), default="protocol")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fault", choices=("none", "early_ready", "drop_signal", "drop_ack", "corrupt_payload"), default="none")
    parser.add_argument("--helper-root", type=Path, default=default_helper_root())
    parser.add_argument("--output-dir", type=Path, default=Path("output/distributed"))
    parser.add_argument("--consumer-delay", type=int, default=7, help="unitless logical ticks")
    parser.add_argument("--segment-timeout", type=float, default=120)
    args = vars(parser.parse_args(argv))
    args["fault"] = None if args["fault"] == "none" else args["fault"]
    try:
        report = run(**args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['run_dir']}/manifest.json", flush=True)
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
