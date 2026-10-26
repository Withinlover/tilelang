"""Isolate Torch NPU startup from TileLang/SHMEM bridge loading (Linux).

Only the standard library is imported by the supervisor. Every device/mode
uses a fresh interpreter with the caller's environment, and runs sequentially.
This does not initialize SHMEM or test communication. Raw reports stay local.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback

MODES = ("plain", "bridge-first", "torch-first")
LIBRARIES = (
    "libascendcl",
    "libruntime",
    "libacl_rt",
    "libascend_hal",
    "libtsdclient",
    "libshmem",
    "libhcomm",
    "libhccl",
    "libtorch_npu",
    "libtilelang",
)
ENVIRONMENT = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "SHMEM_ROOT",
    "SHMEM_LIBRARY",
    "CXX",
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def loaded_libraries():
    """Read mapped paths, without invoking ldd or loading additional libraries."""
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return []
    paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith(LIBRARIES):
            paths.add(fields[5])
    return sorted(paths)


def module_info():
    return {
        name: {"version": str(getattr(module, "__version__", "unknown")), "file": getattr(module, "__file__", None)}
        for name in ("torch", "torch_npu", "tilelang")
        if (module := sys.modules.get(name)) is not None
    }


def probe(mode, device, report_path, *, shmem_root=None, shmem_library=None):
    started = time.monotonic()
    report = {"mode": mode, "device": device, "pid": os.getpid(), "status": "running", "phase": "start", "checkpoints": []}

    def checkpoint(phase):
        report.update(phase=phase, modules=module_info(), elapsed_seconds=round(time.monotonic() - started, 3))
        report["checkpoints"].append(
            {
                "phase": phase,
                "status": report["status"],
                "libraries": loaded_libraries(),
                "torch_npu_imported": "torch_npu" in sys.modules,
            }
        )
        write_json(report_path, report)
        print(f"[{mode} device={device}] {phase}: {report['status']}", flush=True)

    def load_bridge():
        checkpoint("bridge_config")
        if not shmem_root:
            raise ValueError("set SHMEM_ROOT or pass --shmem-root for bridge probes")
        checkpoint("bridge_import")
        # Keep the plain baseline independent of TileLang and its native libs.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        runtime = importlib.import_module("tilelang.ascend.distributed.native_runtime")
        checkpoint("bridge_load")
        bridge = runtime.ShmemBridge(shmem_root=shmem_root, shmem_library=shmem_library)
        report["bridge"] = {"library": str(bridge.lib._name), "fingerprint": bridge.fingerprint}
        checkpoint("bridge_loaded")
        return bridge

    def compute(torch, prefix):
        checkpoint(f"{prefix}.tensor_create")
        x = torch.tensor([1.0, 2.0], dtype=torch.float32, device=f"npu:{device}")
        checkpoint(f"{prefix}.tensor_add")
        y = x + x
        checkpoint(f"{prefix}.synchronize")
        torch.npu.synchronize()
        checkpoint(f"{prefix}.copy_to_cpu")
        actual = y.cpu().tolist()
        checkpoint(f"{prefix}.verify")
        if actual != [2.0, 4.0]:
            raise RuntimeError(f"incorrect NPU addition result: {actual!r}; expected [2.0, 4.0]")
        report.setdefault("results", {})[prefix] = actual

    try:
        checkpoint("start")
        bridge = load_bridge() if mode == "bridge-first" else None
        checkpoint("import_torch")
        torch = importlib.import_module("torch")
        checkpoint("import_torch_npu")
        importlib.import_module("torch_npu")
        checkpoint("set_device")
        torch.npu.set_device(device)
        compute(torch, "before_bridge" if mode == "torch-first" else "compute")
        if mode == "torch-first":
            bridge = load_bridge()
            compute(torch, "after_bridge")
        if bridge is not None:
            checkpoint("bridge_device_identity")
            report["device_identity"] = bridge.device_identity(device)
        report["status"] = "passed"
        checkpoint("complete")
        return 0
    except Exception as error:
        report["status"] = "failed"
        detail = traceback.format_exc()
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": detail}
        # Compiler/linker stderr is otherwise hidden by capture_output=True.
        if isinstance(error, subprocess.CalledProcessError):
            for key in ("stdout", "stderr"):
                value = getattr(error, key, None)
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                report["error"][key] = value
                if value:
                    print(value, file=sys.stderr, flush=True)
        checkpoint(report["phase"])
        print(detail, file=sys.stderr, flush=True)
        return 1


def stop_probe(process):
    """Also kill compiler descendants if the probe leader exits on SIGTERM."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def run_probe(command, report_path, log_path, *, mode, device, timeout):
    started = time.monotonic()
    interrupted = False
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_probe(process)
        except KeyboardInterrupt:
            interrupted = True
            stop_probe(process)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {"mode": mode, "device": device, "phase": "process_start", "status": "incomplete"}
    # Even a child that wrote 'passed' can hang/crash during native teardown.
    if timed_out or interrupted:
        report["status"] = "interrupted" if interrupted else "timeout"
    elif process.returncode != 0 or report.get("status") != "passed":
        report["status"] = "failed"
    report.update(
        returncode=process.returncode,
        timed_out=timed_out,
        elapsed_seconds=round(time.monotonic() - started, 3),
        log=str(log_path),
    )
    # Native stderr can contain error codes absent from the Python exception.
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    # Bare six-digit strings may be PIDs, timestamps or private path components.
    numeric_codes = re.findall(r"\b(?:error[_ ]?code|return[_ ]?code)\s*(?:is\s*|[=:]\s*)?(\d{6})\b", log_text, re.IGNORECASE)
    report["error_codes"] = sorted(set(numeric_codes + re.findall(r"\b[A-Z]{2,3}\d{4,5}\b", log_text)))
    write_json(report_path, report)
    return report


def compact_summary(report):
    """Allowlist for manual review; not a guarantee of confidentiality."""
    probes = []
    for item in report["probes"]:
        entry = {
            key: item[key] for key in ("mode", "device", "status", "phase", "returncode", "timed_out", "elapsed_seconds", "error_codes")
        }
        entry["versions"] = {name: info["version"] for name, info in item.get("modules", {}).items()}
        entry["error_type"] = item.get("error", {}).get("type")
        entry["library_checkpoints"] = [
            {
                "phase": checkpoint["phase"],
                "torch_npu_imported": checkpoint["torch_npu_imported"],
                "libraries": sorted({Path(path).name for path in checkpoint["libraries"]}),
            }
            for checkpoint in item.get("checkpoints", [])
        ]
        probes.append(entry)
    return {
        "schema_version": 1,
        "passed": report["passed"],
        "scope": "single-device startup and bridge loading only; no SHMEM initialization or communication",
        "notice": "Review and redact before sharing. Full paths, environment and tracebacks are in local raw reports only.",
        "probes": probes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="logical NPU indices under the current visibility mask")
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--shmem-root", default=os.environ.get("SHMEM_ROOT"))
    parser.add_argument("--shmem-library", default=os.environ.get("SHMEM_LIBRARY"))
    parser.add_argument("--timeout", type=float, default=180, help="seconds per fresh process, including bridge compilation")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/distributed/diagnostics"), help="parent directory for a new run directory"
    )
    parser.add_argument("--_worker-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error("this diagnostic requires Linux (/proc and process group cleanup)")
    if any(device < 0 for device in args.devices) or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("devices must be nonnegative and timeout must be positive and finite")
    if args._worker_report:
        if len(args.devices) != 1 or len(args.modes) != 1:
            parser.error("internal worker requires exactly one device and mode")
        return probe(args.modes[0], args.devices[0], args._worker_report, shmem_root=args.shmem_root, shmem_library=args.shmem_library)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir.resolve()))
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": sys.version},
        "environment": {name: os.environ[name] for name in ENVIRONMENT if name in os.environ},
        "options": {
            "devices": args.devices,
            "modes": args.modes,
            "timeout": args.timeout,
            "shmem_root": args.shmem_root,
            "shmem_library": args.shmem_library,
        },
        "passed": False,
        "probes": [],
    }
    print(f"Results: {directory}", flush=True)
    interrupted = False
    for device in dict.fromkeys(args.devices):
        for mode in dict.fromkeys(args.modes):
            report_path = directory / f"device{device}-{mode}.json"
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--_worker-report",
                str(report_path),
                "--devices",
                str(device),
                "--modes",
                mode,
            ]
            for key, value in (("--shmem-root", args.shmem_root), ("--shmem-library", args.shmem_library)):
                if value:
                    command.extend([key, value])
            print(f"Running device={device} mode={mode} (timeout={args.timeout:g}s)", flush=True)
            result = run_probe(command, report_path, report_path.with_suffix(".log"), mode=mode, device=device, timeout=args.timeout)
            report["probes"].append(result)
            print(f"  {result['status']}: {result['phase']} (exit={result['returncode']})", flush=True)
            write_json(directory / "report.json", report)
            write_json(directory / "summary.json", compact_summary(report))
            interrupted = result["status"] == "interrupted"
            if interrupted:
                break
        if interrupted:
            break
    report["passed"] = not interrupted and all(item["status"] == "passed" for item in report["probes"])
    write_json(directory / "report.json", report)
    write_json(directory / "summary.json", compact_summary(report))
    print("Review/redact summary.json before sharing; keep raw reports/logs local.", flush=True)
    return 130 if interrupted else (0 if report["passed"] else 1)


if __name__ == "__main__":
    sys.exit(main())
