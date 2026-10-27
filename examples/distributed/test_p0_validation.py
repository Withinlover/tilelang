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
        assert command[command.index("--execution-policy") + 1] == "artifact"
        directory.mkdir(parents=True, exist_ok=True)
        runner.write_json(
            directory / "summary.json",
            performance_evidence(iterations=20),
        )
        # A launcher can finish normally at the deadline after writing output.
        return {"exit_code": 0, "timed_out": timed_out}

    monkeypatch.setattr(runner, "run_command", finished_worker)
    result = runner.main(["--stages", "performance", "--output-dir", str(tmp_path), "--shmem-root", str(runner.ROOT)])
    report = json.loads(next(tmp_path.glob("run_*/detail.json")).read_text())
    compact = json.loads(next(tmp_path.glob("run_*/summary.json")).read_text())
    assert result == int(timed_out)
    assert report["passed"] is not timed_out
    assert ("performance_comparison" in report) is not timed_out
    assert report["device_acceptance"] == "not_run"
    assert compact["reported_device_acceptance"] == "not_run"
    assert "results" not in compact and "passed" not in compact
    assert assess_performance({"exit_code": 0, "timed_out": False}, compact, iterations=20)["status"] == "failed"


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


@pytest.mark.parametrize(
    "mode,policy,mixed_guard", [("hccl", None, False), ("p0b", "artifact", False), ("p0b", "strict", False), ("p0b", "artifact", True)]
)
def test_benchmark_main_preserves_timing_boundaries_and_nested_attribution(tmp_path, monkeypatch, mode, policy, mixed_guard):
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
        copies = [{**deepcopy(value), "rank": rank} for rank in range(2)]
        if mixed_guard:
            copies[1]["source_validation"] = "serialized"
        return copies

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

        def compile_program(*args, **kwargs):
            assert kwargs["execution_policy"] == policy
            return compiled

        monkeypatch.setattr(benchmark.D, "compile", compile_program)
        monkeypatch.setattr(benchmark.D, "init", lambda **kwargs: session)
        monkeypatch.setattr(benchmark.D, "alloc_symmetric", lambda *args, **kwargs: object())
        monkeypatch.setattr(benchmark.D, "channel", lambda *args, **kwargs: object())
        monkeypatch.setattr(benchmark.D, "prepare_launch", lambda *args, **kwargs: object())
        monkeypatch.setattr(benchmark.D, "copy_in", lambda *args, **kwargs: None)
        monkeypatch.setattr(benchmark.D, "launch", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            benchmark.D,
            "complete",
            lambda _: {
                "physical_kernel_count": 1,
                "execution_policy": policy,
                "source_validation": "identity" if policy == "artifact" else "serialized",
                "timing_basis": "host_wall",
                "timing_seconds": {"plan_validation": 1e-9},
            },
        )
        monkeypatch.setattr(benchmark.D, "close", lambda _: None)
    argv = ["--mode", mode, "--warmup", "1", "--iterations", "2", "--output-dir", str(tmp_path)]
    if policy == "strict":
        argv += ["--execution-policy", policy]
    if mixed_guard:
        with pytest.raises(AssertionError, match="different source validation"):
            benchmark.main(argv)
        assert not (tmp_path / "summary.json").exists()
        return
    assert benchmark.main(argv) == 0
    summary = json.loads((tmp_path / "detail.json").read_text())
    compact = json.loads((tmp_path / "summary.json").read_text())
    assert "ranks" not in compact and "passed" not in compact
    assert summary["execution_policy"] == compact["performance"][mode]["execution_policy"] == policy
    assert (
        summary["source_validation"]
        == compact["performance"][mode]["source_validation"]
        == (None if mode == "hccl" else "identity" if policy == "artifact" else "serialized")
    )
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


@pytest.mark.parametrize("schema", [1, 2])
def test_offline_report_preserves_metrics_filters_private_data_and_is_bounded(tmp_path, schema):
    from examples.distributed.summarize_p0 import compact_report, main as summarize, render_markdown

    private = "private_account.private_hostname"
    report = {"schema_version": 1, "passed": True, "device_acceptance": "not_run", "requested_stages": ["performance"], "results": []}
    for mode, factor in (("hccl", 1), ("p0a", 10), ("p0b", 100)):
        evidence = performance_evidence(iterations=3)
        evidence.update(
            schema_version=schema,
            mode=mode,
            rounds=3,
            slots=2,
            tile=64,
            dtype="bfloat16",
            output_dtype="float32",
            first_build_or_cache_load_seconds=[1.0, 2.0],
            first_execution_seconds=[0.02, 0.03],
            first_end_to_end_seconds=[0.04, 0.05],
        )
        for key in ("execution", "end_to_end"):
            evidence[key] = {"samples_seconds": [factor * value for value in (0.001, 0.002, 0.003)], "median_seconds": factor * 0.002}
        if schema == 1:
            ranks = evidence.pop("ranks")
            directory = tmp_path / "performance" / mode
            directory.mkdir(parents=True)
            for rank in ranks:
                # Schema 1 did not yet record per-iteration passed/mismatch fields.
                for check in rank["checks"]:
                    check.pop("passed")
                    check.pop("mismatches")
                    check.pop("nonfinite")
                (directory / f"rank{rank['rank']}.json").write_text(json.dumps(rank))
        else:
            evidence["stages"] = {"prepare": {"median_seconds": 0.004}, private: {"median_seconds": 123.0}}
            evidence["native_run"] = {"plan_validation": {"median_seconds": 0.005}, private: {"median_seconds": 123.0}}
        report["results"].append(
            {
                "stage": "performance",
                "mode": mode,
                "status": "passed",
                "summary": evidence,
                "process": {"exit_code": 0, "timed_out": False, "command": ["/home/" + private]},
                "manifest": str(tmp_path / "must_not_read.json"),
                "error": "RuntimeError: " + private,
            }
        )
    source = tmp_path / "summary.json"
    original = json.dumps(report)
    source.write_text(original)
    (tmp_path / "must_not_read.json").write_text("not JSON: " + private)
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "commit": {"exit_code": 0, "output": "a" * 40},
                "packages": {"torch": "2.8.0.post1", "torch_npu": "2.8.0+" + private},
                "status": {"output": "/home/" + private},
            }
        )
    )
    out = tmp_path / "shareable"
    assert summarize([str(source), "--output-dir", str(out)]) == 0
    first = (out / "compact.json").read_text()
    markdown = (out / "report.md").read_text()
    compact = json.loads(first)
    assert compact["commit"] == "a" * 40
    assert compact["versions"]["torch"] == "2.8.0.post1" and compact["versions"]["torch_npu"] is None
    assert compact["new_device_execution"] is False and compact["reported_device_acceptance"] == "not_run"
    value = compact["performance"]["p0a"]
    assert value["execution"] == {"samples": 3, "median_ms": 20.0, "p95_ms": pytest.approx(29.0)}
    assert value["numerics"]["max_abs_error"] == 1e-7
    assert value["first_rank_max_ms"]["first_build_or_cache_load"] == 2000.0
    assert value["ratios"]["execution"] == {"hccl_over_mode": 0.1, "slowdown_x": 10.0}
    assert value["stages_median_ms"]["prepare"] == (None if schema == 1 else 4.0)
    assert value["native_median_ms"]["plan_validation"] == (None if schema == 1 else 5.0)
    assert value["stage_availability"] == ("legacy_no_stage_timings" if schema == 1 else "available")
    assert value["execution_policy"] is None
    assert value["source_validation"] is None
    assert private not in first + markdown and "/home/" not in first + markdown and "samples_seconds" not in first
    assert len(first.encode()) < 16000 and 30 <= len(markdown.splitlines()) <= 60
    assert summarize([str(source), "--output-dir", str(out)]) == 0
    assert (out / "compact.json").read_text() == first and source.read_text() == original
    # A shareable-only copy remains exportable, but can never pass assessment.
    assert summarize([str(out / "compact.json"), "--output-dir", str(tmp_path / "reshare")]) == 0
    reexport = json.loads((tmp_path / "reshare/compact.json").read_text())
    assert reexport["reported_device_acceptance"] == "not_run" and reexport["new_device_execution"] is False
    assert assess_performance({"exit_code": 0, "timed_out": False}, reexport, iterations=3)["status"] == "failed"
    # Increasing raw iterations changes only bounded scalar fields in the export.
    large = deepcopy(report)
    for item in large["results"]:
        item["summary"]["execution"]["samples_seconds"] *= 1000
        item["summary"]["end_to_end"]["samples_seconds"] *= 1000
        for rank in item["summary"].get("ranks", []):
            rank["checks"] *= 1000
    large_compact = compact_report(large)
    assert len(json.dumps(large_compact)) < 16000
    assert len(render_markdown(large_compact).splitlines()) == len(markdown.splitlines())


def test_report_handles_missing_extreme_values_and_names_failed_cases():
    from examples.distributed.summarize_p0 import compact_report, render_markdown

    source = {
        "schema_version": 1,
        "passed": False,
        "error": "ValueError: /home/private_user/private_host",
        "results": [
            {
                "stage": "device",
                "status": "failed",
                "case": {"name": "p0b-hot-slot", "rounds": 7, "slots": 1, "launches": 3},
                "process": {"exit_code": -9, "timed_out": True},
                "rank_exit_codes": [0, -9],
            },
            {"stage": "performance", "mode": "hccl", "summary": {"execution": {"median_seconds": 1e99}}},
            {
                "stage": "performance",
                "mode": "p0b",
                "summary": {
                    "execution": {"median_seconds": 5e-324},
                    "rounds": 10**400,
                    "stages": {"prepare": {"median_seconds": float("nan")}},
                },
            },
        ],
    }
    compact = compact_report(source, provenance={"commit": "abcdef0"})
    text = json.dumps(compact, allow_nan=False) + render_markdown(compact)
    assert compact["commit"] is None
    assert compact["performance"]["p0b"]["parameters"]["rounds"] is None
    assert compact["performance"]["p0b"]["stages_median_ms"]["prepare"] is None
    assert compact["performance"]["p0b"]["ratios"]["execution"]["hccl_over_mode"] is None
    assert compact["performance"]["p0a"] is None
    assert "p0b-hot-slot exit=-9,timeout=true" in text and "ValueError" in text
    assert "private_user" not in text and "private_host" not in text


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


@pytest.mark.parametrize("policy", ["artifact", "strict"])
def test_validation_matrix_forwards_policy_and_records_case_metadata(tmp_path, monkeypatch, policy):
    from examples.distributed import run_p0_validation as runner
    from examples.distributed.summarize_p0 import compact_report

    args = SimpleNamespace(
        execution_policy=policy,
        sdk_profile="native",
        timeout=900,
        operation_timeout=120,
        shmem_root=None,
        shmem_library=None,
        devices=[0, 1],
    )
    monkeypatch.setattr(runner, "run_command", lambda *args, **kwargs: {"exit_code": 2, "timed_out": False})
    results = []
    for case in CASES:
        expected = "strict" if case.execution == "phased" else policy
        command = runner.case_command(args, case, "device", tmp_path / case.name)
        assert command[command.index("--execution-policy") + 1] == expected
        result = runner.run_case(args, case, "device", tmp_path / case.name)
        assert result["case"]["execution_policy"] == expected
        results.append(result)
    compact = compact_report({"schema_version": 1, "results": results})
    assert len(compact["cases"]) == len(CASES)
    for report in (compact, compact_report(compact)):
        for case in report["cases"]:
            assert case["parameters"]["execution_policy"] == ("strict" if case["case"].startswith("phased-") else policy)


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
