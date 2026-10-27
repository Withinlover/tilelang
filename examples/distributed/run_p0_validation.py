"""Supervised P0 acceptance: CPU regression, SDK build, physical devices, timing.

Only a completed device case with two successful workers and independent output
validation counts as device evidence. A successful build is never device PASS.
"""

import argparse
from contextlib import suppress
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Case:
    name: str
    execution: str
    operation: str
    rounds: int = 3
    slots: int = 2
    launches: int = 2
    delay_ms: int = 0
    fault: str = "none"


CASES = (
    Case("phased-put", "phased", "put"),
    Case("phased-get", "phased", "get"),
    Case("phased-gemm", "phased", "gemm"),
    Case("p0a-put", "p0a", "put"),
    Case("p0a-get", "p0a", "get"),
    Case("p0a-vector", "p0a", "vector"),
    Case("p0a-hot-slot", "p0a", "vector", rounds=7, slots=1, launches=3, delay_ms=200),
    Case("p0a-unused-slot", "p0a", "vector", rounds=1, slots=2),
    Case("p0a-negative", "p0a", "vector", fault="corrupt_send"),
    Case("p0b-gemm", "p0b", "gemm"),
    Case("p0b-hot-slot", "p0b", "gemm", rounds=7, slots=1, launches=3, delay_ms=200),
    Case("p0b-unused-slot", "p0b", "gemm", rounds=1, slots=2),
    Case("p0b-negative", "p0b", "gemm", fault="corrupt_send"),
)
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def descendant_groups(pid):
    """Include rank workers which deliberately started independent sessions."""
    groups = {pid}
    if sys.platform != "linux":
        return groups
    entries = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            entries[int(path.name)] = (int(fields[1]), int(fields[2]))
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
    descendants = {pid}
    while True:
        found = {child for child, (parent, _) in entries.items() if parent in descendants} - descendants
        if not found:
            break
        descendants.update(found)
    groups.update(entries[child][1] for child in descendants if child in entries)
    groups.discard(os.getpgrp())
    return groups


def terminate(process, groups=None):
    groups = set(groups or ()) | descendant_groups(process.pid)
    for group in groups:
        with suppress(ProcessLookupError):
            os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + 3
    while groups and time.monotonic() < deadline:
        process.poll()
        for group in tuple(groups):
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                groups.remove(group)
        if groups:
            time.sleep(0.05)
    for group in groups:
        with suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
    process.wait(timeout=5)


def run_command(command, directory, timeout, *, env_overrides=None):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "command.json", command)
    if env_overrides:
        write_json(directory / "environment_overrides.json", env_overrides)
    result = {"command": command, "exit_code": None, "timed_out": False, "elapsed_seconds": None}
    started = time.perf_counter()
    process = None
    groups = set()
    try:
        with (directory / "process.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, **(env_overrides or {})}
            )
            deadline = time.monotonic() + timeout
            while True:
                groups.update(descendant_groups(process.pid))
                result["exit_code"] = process.poll()
                if result["exit_code"] is not None:
                    break
                if time.monotonic() >= deadline:
                    result["timed_out"] = True
                    terminate(process, groups)
                    result["exit_code"] = process.returncode
                    break
                time.sleep(0.05)
            # A failed launcher can exit before its independent worker sessions.
            # Use groups observed while it was alive instead of trusting PPIDs
            # after those workers have already been reparented.
            terminate(process, groups)
    except BaseException:
        if process is not None:
            terminate(process, groups)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(directory / "process.json", result)
    return result


def assess_device(process, manifest, *, negative):
    """Do not accept a crash, timeout, build or unrelated mismatch as a negative."""
    expected_exit = 1 if negative else 0
    workers = manifest.get("workers", [])
    executed = len(workers) == 2 and all(
        worker.get("completed")
        and worker.get("closed")
        and worker.get("launches")
        and all(entry.get("execution", {}).get("device_executed") and entry["execution"].get("completed") for entry in worker["launches"])
        for worker in workers
    )
    accepted = (
        not process["timed_out"]
        and process["exit_code"] == expected_exit
        and manifest.get("rank_exit_codes") == [0, 0]
        and manifest.get("report_complete")
        and executed
        and manifest.get("expected_negative_detected" if negative else "passed")
    )
    return {"status": "passed" if accepted else "failed", "device_executed": bool(executed), "negative_expected": negative}


def case_command(args, case, stage, directory):
    command = [
        sys.executable,
        "-m",
        "examples.distributed.run_native_distributed",
        "--mode",
        "build" if stage == "compile" else "run",
        "--execution",
        case.execution,
        "--case",
        case.operation,
        "--rounds",
        str(case.rounds),
        "--slots",
        str(case.slots),
        "--launches",
        str(case.launches),
        "--output-dir",
        str(directory / "artifacts"),
        "--sdk-profile",
        args.sdk_profile,
        "--timeout",
        str(max(1, args.timeout - 15)),
        "--operation-timeout",
        str(args.operation_timeout),
        "--rank-delay-ms",
        str(case.delay_ms),
        "--fault",
        case.fault,
        "--devices",
        *map(str, args.devices),
    ]
    if args.shmem_root:
        command += ["--shmem-root", args.shmem_root]
    if args.shmem_library:
        command += ["--shmem-library", args.shmem_library]
    return command


def run_case(args, case, stage, directory):
    process = run_command(case_command(args, case, stage, directory), directory, args.timeout)
    result = {
        "stage": stage,
        "case": asdict(case),
        "directory": str(directory),
        "process": process,
        "status": "failed",
        "device_executed": False,
    }
    if stage == "compile":
        path = directory / "artifacts/validation.json"
        if path.is_file():
            build = json.loads(path.read_text())
            result["status"] = (
                "passed"
                if process["exit_code"] == 0
                and not process["timed_out"]
                and build.get("compiled")
                and build.get("device_executed") is False
                else "failed"
            )
        return result
    paths = list((directory / "artifacts").glob("p0_native_*/manifest.json"))
    if len(paths) == 1:
        manifest = json.loads(paths[0].read_text())
        result.update(assess_device(process, manifest, negative=case.fault != "none"))
        result["manifest"] = str(paths[0])
        result["rank_exit_codes"] = manifest.get("rank_exit_codes")
        result["accuracy"] = manifest.get("accuracy")
        result["errors"] = manifest.get("errors")
        result["rank_errors"] = [
            {"rank": worker.get("rank"), "error": worker.get("error"), "session_state": worker.get("session_state")}
            for worker in manifest.get("workers", [])
            if worker.get("error")
        ]
    return result


def cpu_regression(args, directory):
    junit = directory / "pytest.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-k",
        "not real_sdk and not real_bridge_dlpack",
        "--junitxml",
        str(junit),
        "testing/ascend/distributed",
        "testing/python/language/test_tilelang_language_distributed.py",
        "testing/python/analysis/test_distributed_checker.py",
        "testing/python/analysis/test_distributed_native_gate.py",
        "testing/python/runtime/test_tilelang_distributed_session.py",
        "examples/distributed/test_native_distributed.py",
        "examples/distributed/test_p0_validation.py",
    ]
    process = run_command(command, directory, args.timeout)
    counts = dict(tests=0, failures=0, errors=0, skipped=0)
    if junit.is_file():
        for suite in ET.parse(junit).getroot().iter("testsuite"):
            for key in counts:
                counts[key] += int(suite.get(key, "0"))
    passed = process["exit_code"] == 0 and not process["timed_out"] and counts["tests"] > counts["skipped"]
    return {"stage": "cpu", "status": "passed" if passed else "failed", "device_executed": False, "process": process, "pytest": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="cpu", help="Comma-separated cpu,compile,device,performance; default cpu")
    parser.add_argument("--cases", default="all", help="Comma-separated case names; use --list")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", default="output/distributed/p0-acceptance")
    parser.add_argument("--sdk-profile", choices=("native", "cann-9.1"), default="cann-9.1")
    parser.add_argument("--shmem-root", default=os.environ.get("SHMEM_ROOT"))
    parser.add_argument("--shmem-library", default=os.environ.get("SHMEM_LIBRARY"))
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--operation-timeout", type=int, default=120)
    parser.add_argument("--keep-going", action="store_true", help="Continue independent cases after a failure")
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([asdict(case) for case in CASES], indent=2))
        return 0
    stages = args.stages.split(",")
    if not stages or any(stage not in ("cpu", "compile", "device", "performance") for stage in stages) or len(set(stages)) != len(stages):
        parser.error("stages must be unique members of cpu,compile,device,performance")
    names = {case.name for case in CASES} if args.cases == "all" else set(args.cases.split(","))
    if names - {case.name for case in CASES}:
        parser.error(f"unknown cases: {sorted(names - {case.name for case in CASES})}")
    if len(set(args.devices)) != 2 or min(args.devices) < 0:
        parser.error("select two distinct nonnegative device IDs")
    if args.timeout <= 15 or args.operation_timeout <= 0 or args.benchmark_iterations <= 0:
        parser.error("timeout must exceed 15 seconds; operation timeout and benchmark iterations must be positive")
    if any(stage != "cpu" for stage in stages) and not args.shmem_root:
        parser.error("compile/device/performance requires SHMEM_ROOT or --shmem-root")
    base = Path(args.output_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run_", dir=base))
    print(f"Acceptance directory: {root}", flush=True)
    provenance = {"python_version": sys.version, "packages": {}}
    for package in ("tilelang", "torch", "torch_npu", "apache-tvm-ffi", "pytest"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = None
    for key, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
        ("submodules", ["git", "submodule", "status"]),
    ):
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        provenance[key] = {"exit_code": process.returncode, "output": process.stdout.strip()}
    if args.shmem_root:
        process = subprocess.run(["git", "-C", args.shmem_root, "rev-parse", "HEAD"], text=True, capture_output=True)
        provenance["shmem_revision"] = {"exit_code": process.returncode, "output": process.stdout.strip()}
    write_json(root / "provenance.json", provenance)
    report = {"schema_version": 1, "requested_stages": stages, "results": [], "passed": False, "device_acceptance": "not_run"}
    try:
        for stage in stages:
            if stage == "cpu":
                results = [cpu_regression(args, root / "cpu")]
            elif stage == "performance":
                results = []
                for mode in ("hccl", "p0a", "p0b"):
                    directory = root / "performance" / mode
                    with socket.socket() as bootstrap_socket, socket.socket() as control_socket:
                        bootstrap_socket.bind(("127.0.0.1", 0))
                        control_socket.bind(("127.0.0.1", 0))
                        bootstrap_port = bootstrap_socket.getsockname()[1]
                        control_port = control_socket.getsockname()[1]
                    command = [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc_per_node=2",
                        "--log-dir",
                        str(directory / "logs"),
                        "--redirects",
                        "3",
                        "-m",
                        "examples.distributed.benchmark_p0",
                        "--mode",
                        mode,
                        "--iterations",
                        str(args.benchmark_iterations),
                        "--bootstrap-port",
                        str(bootstrap_port),
                        "--control-port",
                        str(control_port),
                        "--output-dir",
                        str(directory),
                        "--sdk-profile",
                        args.sdk_profile,
                        "--shmem-root",
                        args.shmem_root,
                        "--devices",
                        *map(str, args.devices),
                    ]
                    if args.shmem_library:
                        command += ["--shmem-library", args.shmem_library]
                    process = run_command(
                        command, directory, args.timeout, env_overrides={"P0_BENCH_CACHE_ROOT": str(directory / "cold-cache")}
                    )
                    path = directory / "summary.json"
                    summary = json.loads(path.read_text()) if path.is_file() else {}
                    results.append(
                        {
                            "stage": stage,
                            "mode": mode,
                            "process": process,
                            "status": "passed"
                            if process["exit_code"] == 0
                            and not process["timed_out"]
                            and summary.get("passed")
                            and summary.get("device_executed")
                            else "failed",
                            "device_executed": bool(summary.get("device_executed")),
                            "summary": summary,
                        }
                    )
            else:
                results = []
                for case in CASES:
                    if case.name not in names:
                        continue
                    result = run_case(args, case, stage, root / stage / case.name)
                    results.append(result)
                    report["results"] = report["results"] + [result]
                    write_json(root / "summary.json", report)
                    print(f"{stage}/{case.name}: {result['status']}", flush=True)
                    if result["status"] != "passed" and not args.keep_going:
                        break
                # Results have already been persisted individually.
                results = []
            report["results"].extend(results)
            write_json(root / "summary.json", report)
            if any(result["status"] != "passed" for result in report["results"]) and not args.keep_going:
                break
        performance = {item["mode"]: item for item in report["results"] if item["stage"] == "performance"}
        if set(performance) == {"hccl", "p0a", "p0b"} and all(item["status"] == "passed" for item in performance.values()):
            report["performance_comparison"] = {
                boundary: {
                    mode: performance["hccl"]["summary"][boundary]["median_seconds"]
                    / performance[mode]["summary"][boundary]["median_seconds"]
                    for mode in ("p0a", "p0b")
                }
                for boundary in ("execution", "end_to_end")
            }
            report["performance_ratio_definition"] = "HCCL baseline median / candidate median; > 1 means lower candidate wall time"
        report["passed"] = all(result["status"] == "passed" for result in report["results"]) and bool(report["results"])
        devices = [result for result in report["results"] if result["stage"] == "device"]
        if devices:
            report["device_acceptance"] = (
                "passed_requested_cases"
                if len(devices) == len(names) and all(result["status"] == "passed" for result in devices)
                else "failed_or_incomplete"
            )
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(root / "summary.json", report)
    print(
        json.dumps({"passed": report["passed"], "device_acceptance": report["device_acceptance"], "summary": str(root / "summary.json")}),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
