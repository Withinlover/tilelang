"""Acceptance rejects partial device evidence and preserves real process failures."""

from contextlib import suppress
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from examples.distributed.run_p0_validation import CASES, assess_device, assess_performance, run_command


def performance_evidence(iterations=2):
    return {
        "schema_version": 2,
        "passed": True,
        "device_executed": True,
        "correctness_scope": "benchmark_workload_only",
        "iterations": iterations,
        "warmup": 1,
        "ranks": [
            {
                "rank": rank,
                "passed": True,
                "device_executed": True,
                "checks": [
                    {"iteration": i, "passed": True, "mismatches": 0, "nonfinite": 0, "max_abs_error": 1e-7} for i in range(iterations + 1)
                ],
            }
            for rank in range(2)
        ],
        **{
            boundary: {
                "median_seconds": value,
                "p90_seconds": value,
                "p95_seconds": value,
                "count": iterations,
                "samples_seconds": [value] * iterations,
            }
            for boundary, value in (("execution", 1.0), ("end_to_end", 2.0))
        },
    }


@pytest.mark.parametrize("timed_out", [False, True])
def test_performance_deadline_overrides_successful_worker_summary(tmp_path, monkeypatch, timed_out):
    from examples.distributed import run_p0_validation as runner

    def finished_worker(command, directory, timeout, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        runner.write_json(
            directory / "summary.json",
            performance_evidence(iterations=20),
        )
        # A launcher can finish normally at the deadline after writing output.
        return {"exit_code": 0, "timed_out": timed_out}

    monkeypatch.setattr(runner, "run_command", finished_worker)
    result = runner.main(["--stages", "performance", "--output-dir", str(tmp_path), "--shmem-root", str(runner.ROOT)])
    report = json.loads(next(tmp_path.glob("run_*/summary.json")).read_text())
    assert result == int(timed_out)
    assert report["passed"] is not timed_out
    assert ("performance_comparison" in report) is not timed_out
    assert report["device_acceptance"] == "not_run"


@pytest.mark.parametrize("damage", ["missing_rank", "missing_check", "wrong_output", "wrong_scope", "missing_p95", "wrong_count"])
def test_performance_rejects_incomplete_accuracy_or_timing(damage):
    process = {"exit_code": 0, "timed_out": False}
    summary = performance_evidence()
    assert assess_performance(process, summary, iterations=2)["status"] == "passed"
    if damage == "missing_rank":
        summary["ranks"].pop()
    elif damage == "missing_check":
        summary["ranks"][1]["checks"].pop()
    elif damage == "wrong_output":
        summary["ranks"][1]["checks"][1].update(passed=False, mismatches=1)
    elif damage == "wrong_scope":
        summary["correctness_scope"] = "full_device_acceptance"
    elif damage == "missing_p95":
        del summary["execution"]["p95_seconds"]
    else:
        summary["end_to_end"]["samples_seconds"].pop()
    assert assess_performance(process, summary, iterations=2)["status"] == "failed"


def test_benchmark_pairs_iterations_before_percentiles_and_rejects_partial_ranks():
    from examples.distributed.benchmark_p0 import paired_summary

    reports = [
        {
            "rank": rank,
            "timing_samples": [
                {
                    "iteration": iteration,
                    "warmup": False,
                    "execution_seconds": value,
                    "end_to_end_seconds": value + 1,
                    "stages_seconds": {"launch": value},
                    "native_run_seconds": {"plan_validation": value / 2},
                }
                for iteration, value in enumerate(values)
            ],
        }
        for rank, values in enumerate(([1.0, 100.0], [80.0, 2.0]))
    ]
    result = paired_summary(reports, iterations=2)
    assert result["execution"]["samples_seconds"] == [80.0, 100.0]
    assert result["execution"]["median_seconds"] == 90.0
    assert result["execution"]["p90_seconds"] == 98.0
    assert result["execution"]["p95_seconds"] == 99.0
    assert result["stages"]["launch"]["median_seconds"] == 90.0
    assert result["native_run"]["plan_validation"]["median_seconds"] == 45.0
    reports[1]["timing_samples"].pop()
    with pytest.raises(ValueError, match="incomplete steady-state"):
        paired_summary(reports, iterations=2)


@pytest.mark.parametrize("values", [[], [float("nan")], [-1.0]])
def test_benchmark_rejects_invalid_timing_samples(values):
    from examples.distributed.benchmark_p0 import describe_samples

    with pytest.raises(ValueError, match="nonempty, finite and nonnegative"):
        describe_samples(values)


def test_benchmark_independent_oracle_records_wrong_or_nonfinite_elements():
    import numpy as np

    from examples.distributed.benchmark_p0 import check_output

    # BF16 1.0 times BF16 1.0 sums to 64, independent of receive data.
    a = np.full((2, 1, 64, 64), 0x3F80, dtype=np.uint16)
    b = np.full((64, 64), 0x3F80, dtype=np.uint16)
    output = np.full((1, 2, 64, 64), 64.0, dtype=np.float32)
    assert check_output(output, a, b)["passed"]
    output[0, 1, 0, 0] = 65.0
    check = check_output(output, a, b)
    assert not check["passed"] and check["mismatches"] == 1 and check["max_abs_error"] == 1.0
    output[0, 1, 0, 0] = np.inf
    check = check_output(output, a, b)
    assert not check["passed"] and check["mismatches"] == check["nonfinite"] == 1


@pytest.mark.parametrize("mode", ["hccl", "p0b"])
def test_benchmark_main_preserves_timing_boundaries_and_nested_attribution(tmp_path, monkeypatch, mode):
    import numpy as np
    import torch
    import torch.distributed as dist

    from examples.distributed import benchmark_p0 as benchmark

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _: None, synchronize=lambda _: None), raising=False)
    original_empty, original_inputs = torch.empty, benchmark.make_inputs
    inputs = {}

    def make_inputs(rounds, seed):
        inputs["a"], inputs["b"] = original_inputs(rounds, seed)
        return inputs["a"], inputs["b"]

    def reports(value):
        return [{**deepcopy(value), "rank": rank} for rank in range(2)]

    monkeypatch.setattr(benchmark, "make_inputs", make_inputs)
    if mode == "hccl":
        monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: original_empty(*args, **{**kwargs, "device": "cpu"}))
        monkeypatch.setattr(benchmark, "local_gemm", lambda: None)
        monkeypatch.setattr(benchmark.tilelang, "compile", lambda *args, **kwargs: lambda a, b, c: c.copy_(a.float() @ b.float()))
        monkeypatch.setattr(dist, "init_process_group", lambda *args, **kwargs: None)
        monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
        monkeypatch.setattr(dist, "barrier", lambda: None)
        monkeypatch.setattr(
            dist,
            "all_gather",
            lambda buffers, source: [
                buffer.copy_(torch.from_numpy(inputs["a"][rank].copy()).view(torch.bfloat16)) for rank, buffer in enumerate(buffers)
            ],
        )
        monkeypatch.setattr(dist, "all_gather_object", lambda destination, value: destination.__setitem__(slice(None), reports(value)))
    else:
        compiled = SimpleNamespace(manifest={"objects": []}, build=lambda **kwargs: None)
        monkeypatch.setattr(benchmark, "allgather_gemm", lambda *args: None)
        for key, shape in (("Send", [3, 64, 64]), ("B", [64, 64]), ("Output", [3, 2, 64, 64])):
            compiled.manifest["objects"].append(
                dict(key=key, shape=shape, dtype="float32" if key == "Output" else "bfloat16", kind="symmetric", remote_access="read_write")
            )

        def copy_out(_):
            a = (inputs["a"].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            b = (inputs["b"].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            return np.swapaxes(a @ b, 0, 1).astype(np.float32)

        session = SimpleNamespace(preflight=lambda _: None, control=SimpleNamespace(all_gather=reports), copy_out=copy_out)
        monkeypatch.setattr(benchmark.D, "compile", lambda *args, **kwargs: compiled)
        monkeypatch.setattr(benchmark.D, "init", lambda **kwargs: session)
        monkeypatch.setattr(benchmark.D, "alloc_symmetric", lambda *args, **kwargs: object())
        monkeypatch.setattr(benchmark.D, "channel", lambda *args, **kwargs: object())
        monkeypatch.setattr(benchmark.D, "prepare_launch", lambda *args, **kwargs: object())
        monkeypatch.setattr(benchmark.D, "copy_in", lambda *args, **kwargs: None)
        monkeypatch.setattr(benchmark.D, "launch", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            benchmark.D,
            "complete",
            lambda _: {"physical_kernel_count": 1, "timing_basis": "host_wall", "timing_seconds": {"plan_validation": 1e-9}},
        )
        monkeypatch.setattr(benchmark.D, "close", lambda _: None)
    assert benchmark.main(["--mode", mode, "--warmup", "1", "--iterations", "2", "--output-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["correctness_scope"] == "benchmark_workload_only" and summary["full_device_acceptance"] == "not_assessed"
    assert summary["execution"]["count"] == summary["end_to_end"]["count"] == 2
    assert assess_performance({"exit_code": 0, "timed_out": False}, summary, iterations=2)["status"] == "passed"
    for worker in summary["ranks"]:
        assert len(worker["checks"]) == len(worker["timing_samples"]) == 3
        for sample in worker["timing_samples"]:
            stages = sample["stages_seconds"]
            assert sum(stages.values()) == pytest.approx(sample["end_to_end_seconds"])
            execution = sum(value for key, value in stages.items() if key not in {"prepare", "staging", "staging_sync", "readback"})
            assert execution == pytest.approx(sample["execution_seconds"])
            assert ("collective_submit" in stages) is (mode == "hccl")
            assert ("launch" in stages) is (mode != "hccl")
            assert bool(sample["native_run_seconds"]) is (mode != "hccl")


def device_evidence(*, negative=False):
    process = {"exit_code": 1 if negative else 0, "timed_out": False}
    manifest = {
        "passed": not negative,
        "expected_negative_detected": negative,
        "report_complete": True,
        "rank_exit_codes": [0, 0],
        "workers": [
            {"rank": rank, "completed": True, "closed": True, "launches": [{"execution": {"device_executed": True, "completed": True}}]}
            for rank in range(2)
        ],
    }
    return process, manifest


@pytest.mark.parametrize("negative", [False, True])
def test_complete_device_evidence_requires_both_rank_processes(negative):
    process, manifest = device_evidence(negative=negative)
    assert assess_device(process, manifest, negative=negative)["status"] == "passed"
    manifest["rank_exit_codes"][1] = 2
    assert assess_device(process, manifest, negative=negative)["status"] == "failed"


@pytest.mark.parametrize(
    "damage", ["timeout", "crash", "compile_only", "missing_rank", "missing_launch", "unrelated_failure", "incomplete"]
)
def test_expected_negative_does_not_accept_missing_or_unrelated_evidence(damage):
    process, manifest = device_evidence(negative=True)
    if damage == "timeout":
        process["timed_out"] = True
    elif damage == "crash":
        process["exit_code"] = 2
    elif damage == "compile_only":
        manifest["workers"][0]["launches"][0]["execution"]["device_executed"] = False
    elif damage == "missing_rank":
        manifest["workers"].pop()
    elif damage == "missing_launch":
        manifest["workers"][0]["launches"] = []
    elif damage == "unrelated_failure":
        manifest["expected_negative_detected"] = False
    else:
        manifest["report_complete"] = False
    assert assess_device(process, manifest, negative=True)["status"] == "failed"


def test_build_json_is_not_device_evidence():
    process = {"exit_code": 0, "timed_out": False}
    assert assess_device(process, {"compiled": True, "device_executed": False}, negative=False)["status"] == "failed"


def test_supervisor_preserves_nonzero_exit_and_log(tmp_path):
    result = run_command([sys.executable, "-c", "print('failure marker'); raise SystemExit(7)"], tmp_path, 10)
    assert result["exit_code"] == 7 and not result["timed_out"]
    assert "failure marker" in (tmp_path / "process.log").read_text()
    assert (tmp_path / "command.json").is_file() and (tmp_path / "process.json").is_file()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process groups")
def test_supervisor_timeout_is_a_failure_with_saved_evidence(tmp_path):
    result = run_command([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, 0.1)
    assert result["timed_out"] and result["exit_code"] != 0
    assert (tmp_path / "process.json").is_file()


def test_required_cases_include_hot_slot_multi_launch_delay_and_negative():
    # Assert coverage contracts of the user-facing suite, not internal lowering.
    for execution in ("p0a", "p0b"):
        cases = [case for case in CASES if case.execution == execution]
        assert any(case.slots == 1 and case.rounds > 1 and case.launches > 1 and case.delay_ms > 0 for case in cases)
        assert any(case.fault == "corrupt_send" for case in cases)
        assert any(case.slots > case.rounds for case in cases)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc and independent process sessions")
@pytest.mark.parametrize("leader_exits_first", [False, True])
def test_supervisor_cleans_detached_term_ignoring_rank(tmp_path, leader_exits_first):
    marker = tmp_path / "detached.pid"
    child = (
        "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(60)" % str(marker)
    )
    launcher = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r],start_new_session=True); time.sleep(%s)" % (
        child,
        "0.4" if leader_exits_first else "60",
    )
    try:
        result = run_command([sys.executable, "-c", launcher], tmp_path, 1.0)
        assert marker.is_file()
        pid = int(marker.read_text())
        state = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 3
        while True:
            try:
                ended = state.read_text().rsplit(")", 1)[1].split()[0] == "Z"
            except (FileNotFoundError, ProcessLookupError):
                ended = True
            if ended or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        assert ended
        assert result["timed_out"] is not leader_exits_first
    finally:
        if marker.is_file():
            with suppress(ProcessLookupError):
                os.killpg(int(marker.read_text()), signal.SIGKILL)
