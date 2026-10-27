"""Acceptance rejects partial device evidence and preserves real process failures."""

from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from examples.distributed.run_p0_validation import CASES, assess_device, run_command


@pytest.mark.parametrize("timed_out", [False, True])
def test_performance_deadline_overrides_successful_worker_summary(tmp_path, monkeypatch, timed_out):
    from examples.distributed import run_p0_validation as runner

    def finished_worker(command, directory, timeout, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        runner.write_json(
            directory / "summary.json",
            {
                "passed": True,
                "device_executed": True,
                "execution": {"median_seconds": 1.0},
                "end_to_end": {"median_seconds": 2.0},
            },
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
