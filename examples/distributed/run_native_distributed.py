"""Build or run the P0 source DSL on two physical Ascend 950 devices.

The parent supervises two independent rank processes. Communication is executed
by SHMEM AIV kernels generated from TD statements; no HCCL or CPU payload
transport is used. All reports and synthetic inputs remain in output_dir.
"""

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

import tilelang.ascend.language as T
import tilelang.distributed as D
import tilelang.language.distributed as TD

from examples.distributed.example_allgather_gemm import TILE, allgather_gemm, global_accuracy, make_inputs


def exchange(rounds=3, slots=2, *, get=False):
    """Exercise copy_async, prefix release, get source leases, and slot reuse."""
    if not 1 <= rounds <= 10000 or not 1 <= slots <= 10000:
        raise ValueError("rounds and slots must be in 1..10000")
    group = TD.GroupSpec("world", 2)
    channel = TD.ChannelSpec("comm", group)

    @T.prim_func
    def program(
        Input: TD.SymBuffer((rounds, TILE, TILE), "bfloat16", group=group, remote_access="read"),
        Source: TD.SymBuffer((slots, TILE, TILE), "bfloat16", group=group, remote_access="read"),
        Recv: TD.SymBuffer((slots, TILE, TILE), "bfloat16", group=group, remote_access="write"),
        Ready: TD.SignalBuffer((slots,), group=group),
        Ack: TD.SignalBuffer((slots,), group=group),
        Output: TD.SymBuffer((rounds, TILE, TILE), "bfloat16", group=group, remote_access="write"),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            rank = TD.rank(group)
            peer = 1 - rank
            for iteration in T.serial(rounds):
                slot = iteration % slots
                epoch = epoch0 + iteration + 1
                if iteration >= slots:
                    TD.wait_signal(Ack[slot], epoch - slots, channel=channel, covers=[])
                T.copy(Input[iteration, :, :], Source[slot, :, :])
                if get:
                    TD.signal(TD.remote(Ready, peer)[slot], epoch, channel=channel, release_regions=[Source[slot, :, :]])
                    TD.wait_signal(Ready[slot], epoch, channel=channel, covers=[TD.remote(Source, peer)[slot, :, :]])
                    fetched = TD.copy_async(TD.remote(Source, peer)[slot, :, :], Recv[slot, :, :], channel=channel)
                    TD.wait(fetched, completion="local")
                else:
                    sent = TD.copy_async(Source[slot, :, :], TD.remote(Recv, peer)[slot, :, :], channel=channel)
                    TD.wait(sent, completion="local")
                    TD.fence(channel=channel)
                    TD.signal(TD.remote(Ready, peer)[slot], epoch, channel=channel, release_regions=[])
                    TD.wait(sent, completion="remote")
                    TD.wait_signal(Ready[slot], epoch, channel=channel, covers=[Recv[slot, :, :]])
                T.copy(Recv[slot, :, :], Output[iteration, :, :])
                TD.signal(TD.remote(Ack, peer)[slot], epoch, channel=channel, release_regions=[] if get else [Recv[slot, :, :]])
            for iteration in T.serial(max(0, rounds - slots), rounds):
                TD.wait_signal(Ack[iteration % slots], epoch0 + iteration + 1, channel=channel, covers=[])
            TD.quiet(channel=channel)

    return program


def make_program(case, rounds, slots):
    return allgather_gemm(rounds, slots) if case == "gemm" else exchange(rounds, slots, get=case == "get")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def expected_signals(rounds, slots, base):
    # Unused slots retain their prior value (zero in this fixed-round runner).
    return np.array([base + slot + 1 + (rounds - slot - 1) // slots * slots if slot < rounds else 0 for slot in range(slots)])


def compile_program(args, output):
    return D.compile(
        make_program(args.case, args.rounds, args.slots),
        profile=D.NATIVE_PROFILE,
        mode="native",
        shmem_root=args.shmem_root,
        compile_profile="cann-9.1-gemm64" if args.sdk_profile == "cann-9.1" else "native",
        transport_compile_profile="cann-9.1-shmem" if args.sdk_profile == "cann-9.1" else "native",
        output_dir=output,
    )


def worker(args):
    rank = args.worker_rank
    root = Path(args.output_dir).resolve()
    output = root / f"rank{rank}"
    output.mkdir()
    report = {"rank": rank, "device": args.devices[rank], "completed": False, "launches": []}
    session = None
    try:
        compiled = compile_program(args, output / "kernels")
        session = D.init(
            profile=D.NATIVE_PROFILE,
            rank=rank,
            device=args.devices[rank],
            bootstrap=f"tcp://127.0.0.1:{args.bootstrap_port}",
            control_port=args.control_port,
            shmem_root=args.shmem_root,
            shmem_library=args.shmem_library,
            timeout=args.operation_timeout,
        )
        report["runtime_abi"] = session.abi_fingerprint
        objects = {}
        for spec in compiled.manifest["objects"]:
            common = dict(key=spec["key"], shape=spec["shape"], dtype=spec["dtype"])
            objects[spec["key"]] = (
                D.alloc_signals(session, **common)
                if spec["kind"] == "signal"
                else D.alloc_symmetric(session, remote_access=spec["remote_access"], **common)
            )
        channel = D.channel(session, key="comm")
        for launch in range(args.launches):
            inputs = np.load(root / f"launch{launch}_inputs.npz", allow_pickle=False)
            a, b = inputs["original_a"], inputs["original_b"]
            ticket = D.prepare_launch(
                session,
                compiled,
                channel=channel,
                epochs=args.rounds,
                protocol=f"native-{args.case}-v1",
                parameters={"rounds": args.rounds, "slots": args.slots, "launch": launch, "fault": args.fault},
            )
            send = a[rank].copy()
            if args.fault == "corrupt_send" and rank == 0 and launch == 0:
                # Corrupt only the transmitted/source operand, preserving the
                # parent's immutable original inputs for the global oracle.
                send[0, 0, 0] = np.uint16(0x4080)  # BF16 4.0, outside generated input range.
            D.copy_in(objects["Send" if args.case == "gemm" else "Input"], send, ticket=ticket)
            if args.case == "gemm":
                D.copy_in(objects["B"], b, ticket=ticket)
            destination = objects["Output"]
            sentinel = (
                np.full(destination.shape, np.nan, dtype="<f4") if args.case == "gemm" else np.full(destination.shape, 0xFFFF, dtype="<u2")
            )
            D.copy_in(destination, sentinel, ticket=ticket)
            execution = D.complete(D.launch(compiled, ticket=ticket))
            actual = session.copy_out(destination)
            ready, ack = session.copy_out(objects["Ready"]), session.copy_out(objects["Ack"])
            np.save(output / f"launch{launch}_output.npy", actual)
            expected = expected_signals(args.rounds, args.slots, ticket.base)
            item = {
                "launch": launch,
                "epoch_base": ticket.base,
                "epoch_end": ticket.end,
                "execution": execution,
                "ready": ready.tolist(),
                "ack": ack.tolist(),
                "expected_signals": expected.tolist(),
                "signals_passed": bool(np.array_equal(ready, expected) and np.array_equal(ack, expected)),
            }
            report["launches"].append(item)
            write_json(output / "worker.json", report)
        D.close(session)
        report["closed"] = session.state == "CLOSED"
        report["completed"] = True
        return 0
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["session_state"] = None if session is None else session.state
        traceback.print_exc()
        return 2
    finally:
        write_json(output / "worker.json", report)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_workers(processes):
    # Every worker owns a process group, including any active compiler children.
    groups = {process.pid for process in processes}
    for group in groups:
        with suppress(ProcessLookupError):
            os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + 3
    while groups and time.monotonic() < deadline:
        for process in processes:
            process.poll()  # Reap leaders, even when descendants remain alive.
        for group in tuple(groups):
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                groups.remove(group)
        if groups:
            time.sleep(0.05)
    # A leader may already have exited while a TERM-ignoring grandchild lives.
    for group in groups:
        with suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
    for process in processes:
        process.wait(timeout=5)


def check_accuracy(case, actual, a, b):
    if case == "gemm":
        if actual.dtype != np.dtype("<f4"):
            raise ValueError("native GEMM output must be float32")
        return global_accuracy(actual, a, b)
    expected = a[::-1]
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("native exchange output shape/dtype differs from the global input contract")
    mismatch = actual != expected
    return {
        "passed": bool(not mismatch.any()),
        "elements": int(actual.size),
        "mismatches": int(mismatch.sum()),
        "reference": "bit-exact original BF16 input from peer",
        "ranks": [{"rank": r, "passed": bool(not mismatch[r].any()), "mismatches": int(mismatch[r].sum())} for r in range(2)],
    }


def validate_outputs(root, args, worker_reports):
    """Use original CPU inputs; never derive expected data from receive buffers."""
    accuracy, fault_accuracy = [], []
    for launch in range(args.launches):
        inputs = np.load(root / f"launch{launch}_inputs.npz", allow_pickle=False)
        a, b = inputs["original_a"], inputs["original_b"]
        actual = np.stack([np.load(root / f"rank{r}/launch{launch}_output.npy", allow_pickle=False) for r in range(2)])
        result = check_accuracy(args.case, actual, a, b)
        result["launch"] = launch
        accuracy.append(result)
        if args.fault == "corrupt_send":
            injected = a.copy()
            if launch == 0:
                injected[0, 0, 0, 0] = np.uint16(0x4080)
            faulty = check_accuracy(args.case, actual, injected, b)
            faulty.update(launch=launch, reference="independent oracle with the specified single source corruption")
            fault_accuracy.append(faulty)
    compiled = D.compile(make_program(args.case, args.rounds, args.slots), profile=D.NATIVE_PROFILE)
    complete = len(worker_reports) == 2
    for rank, item in enumerate(worker_reports):
        complete = (
            complete
            and item.get("rank") == rank
            and item.get("completed")
            and item.get("closed")
            and len(item["launches"]) == args.launches
        )
        for launch, entry in enumerate(item["launches"]):
            base, end = launch * args.rounds, (launch + 1) * args.rounds
            plan = compiled.plan(epoch_base=base, epoch_end=end)[rank]
            phases = [{"index": i, "op": instruction[0]} for i, instruction in enumerate(plan)]
            expected = expected_signals(args.rounds, args.slots, base).tolist()
            execution = entry["execution"]
            complete = complete and all(execution.get(key) for key in ("passed", "completed", "device_executed", "kernel_executed"))
            complete = complete and entry["signals_passed"] and entry["ready"] == entry["ack"] == expected
            complete = (
                complete and execution.get("rank") == rank and execution.get("epoch_base") == base and execution.get("epoch_end") == end
            )
            complete = complete and execution.get("phases_count") == len(phases) and execution.get("phases") == phases
    reference_passed = all(item["passed"] for item in accuracy)
    # The negative must complete all native phases and affect rank 1, which
    # consumes rank 0's remote operand; an unrelated crash is never accepted.
    fault_reference_passed = bool(fault_accuracy) and all(item["passed"] for item in fault_accuracy)
    negative_detected = (
        args.fault == "corrupt_send"
        and complete
        and fault_reference_passed
        and not accuracy[0]["ranks"][1]["passed"]
        and all(item["passed"] for item in accuracy[1:])
    )
    return {
        "report_complete": bool(complete),
        "passed": bool(args.fault == "none" and complete and reference_passed),
        "expected_negative_detected": bool(negative_detected),
        "accuracy": accuracy,
        "fault_reference_passed": fault_reference_passed,
        "fault_accuracy": fault_accuracy,
    }


def supervise(args):
    base = Path(args.output_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"p0_native_{args.case}_", dir=base))
    print(f"Native output: {root}", flush=True)
    for launch in range(args.launches):
        a, b = make_inputs(args.rounds, args.seed + launch)
        np.savez(root / f"launch{launch}_inputs.npz", original_a=a, original_b=b)
    args.bootstrap_port = args.bootstrap_port or free_port()
    args.control_port = args.control_port or free_port()
    if args.bootstrap_port == args.control_port:
        raise ValueError("SHMEM bootstrap and control ports must differ")
    report = {
        "profile": D.NATIVE_PROFILE,
        "case": args.case,
        "devices": args.devices,
        "fault": args.fault,
        "rounds": args.rounds,
        "slots": args.slots,
        "launches": args.launches,
        "seed": args.seed,
        "passed": False,
        "report_complete": False,
        "expected_negative_detected": False,
        "execution": "real SHMEM MTE device communication and TileLang kernels",
        "errors": [],
    }
    processes, logs = [], []
    try:
        for rank in range(2):
            command = [
                sys.executable,
                "-m",
                "examples.distributed.run_native_distributed",
                "--mode",
                "run",
                "--worker-rank",
                str(rank),
                "--case",
                args.case,
                "--rounds",
                str(args.rounds),
                "--slots",
                str(args.slots),
                "--launches",
                str(args.launches),
                "--devices",
                *map(str, args.devices),
                "--output-dir",
                str(root),
                "--shmem-root",
                str(args.shmem_root),
                "--sdk-profile",
                args.sdk_profile,
                "--bootstrap-port",
                str(args.bootstrap_port),
                "--control-port",
                str(args.control_port),
                "--operation-timeout",
                str(args.operation_timeout),
                "--fault",
                args.fault,
            ]
            if args.shmem_library:
                command.extend(["--shmem-library", args.shmem_library])
            log = (root / f"rank{rank}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + args.timeout
        while any(process.poll() is None for process in processes):
            failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"rank process failed with exit code(s) {failed}; inspect rank logs")
            if time.monotonic() > deadline:
                raise TimeoutError("two-rank validation exceeded the supervisor timeout")
            time.sleep(0.1)
        if any(process.returncode for process in processes):
            raise RuntimeError("rank process failed; inspect rank logs")
        reports = [json.loads((root / f"rank{rank}/worker.json").read_text()) for rank in range(2)]
        report["workers"] = reports
        report.update(validate_outputs(root, args, reports))
    except (Exception, KeyboardInterrupt) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        stop_workers(processes)
        for log in logs:
            log.close()
        report["rank_exit_codes"] = [process.returncode for process in processes]
        write_json(root / "manifest.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "report_complete", "expected_negative_detected", "errors")}), flush=True)
    return 0 if report["passed"] else (1 if report["expected_negative_detected"] else 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lower", "build", "run"), default="lower")
    parser.add_argument("--case", choices=("gemm", "put", "get"), default="gemm")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--launches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--shmem-root", default=os.environ.get("SHMEM_ROOT"))
    parser.add_argument("--shmem-library", default=os.environ.get("SHMEM_LIBRARY"))
    parser.add_argument("--sdk-profile", choices=("native", "cann-9.1"), default="cann-9.1")
    parser.add_argument("--output-dir", default="output/distributed/native")
    parser.add_argument("--operation-timeout", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--bootstrap-port", type=int)
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--fault", choices=("none", "corrupt_send"), default="none")
    parser.add_argument("--worker-rank", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 10000 or not 1 <= args.slots <= 10000 or not 1 <= args.launches <= 10000 or args.seed < 0:
        parser.error("rounds, slots and launches must be in 1..10000; seed must be nonnegative")
    if len(set(args.devices)) != 2 or min(args.devices) < 0:
        parser.error("select two different nonnegative device IDs")
    if args.operation_timeout <= 0 or args.timeout <= 0:
        parser.error("timeouts must be positive")
    if args.mode != "lower" and not args.shmem_root:
        parser.error("build/run requires --shmem-root or SHMEM_ROOT pointing to the pinned source checkout")
    if args.worker_rank is not None:
        return worker(args)
    if args.mode == "run":
        return supervise(args)
    compiled = compile_program(args, args.output_dir)
    result = compiled.lower(output_dir=args.output_dir, scalars={"epoch0": 0}, epochs=args.rounds, shmem_root=args.shmem_root)
    if args.mode == "build":
        from tilelang.ascend.distributed.native_runtime import build_host_bridge

        compiled.build(shmem_root=args.shmem_root, scalars={"epoch0": 0}, epochs=args.rounds, output_dir=args.output_dir)
        bridge, identity = build_host_bridge(shmem_root=args.shmem_root, shmem_library=args.shmem_library, output_dir=args.output_dir)
        result["host_bridge"] = {"path": str(bridge), **identity}
        result["compiled"] = True
    write_json(Path(args.output_dir) / "validation.json", result)
    print(json.dumps({"mode": args.mode, "device_executed": False, "output_dir": args.output_dir}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
