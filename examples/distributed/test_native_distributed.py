"""Hardware-free checks of native runner plans and independent result acceptance."""

from types import SimpleNamespace
from contextlib import suppress
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

import tilelang.distributed as D
import tilelang.testing

if not tilelang.testing.ascend_backend_compiled():
    pytest.skip("Ascend backend is not compiled", allow_module_level=True)

from examples.distributed.example_allgather_gemm import make_inputs
from examples.distributed.run_native_distributed import expected_signals, make_program, stop_workers, validate_outputs


@pytest.mark.parametrize("case", ["put", "get", "gemm"])
def test_native_examples_plan_both_ranks_without_helper(case):
    compiled = D.compile(make_program(case, 3, 2), profile=D.NATIVE_PROFILE)
    plans = compiled.plan(scalars={"epoch0": 6}, epoch_base=6, epoch_end=9)
    assert set(plans) == {0, 1}
    for rank, plan in plans.items():
        assert plan[-1] == ("quiet",)
        if case == "gemm":
            assert sum(item[0] == "gemm" for item in plan) == 6
            transfers = [item for item in plan if item[0] == "put_signal"]
            assert [item[5] for item in transfers] == [7, 8, 9]
        else:
            transfers = [item for item in plan if item[0] == "copy_async"]
            assert len(transfers) == 3
            assert all(item[2].rank == (1 - rank if case == "get" else rank) for item in transfers)


def _results(tmp_path, case, *, fault="none", corrupt=False):
    args = SimpleNamespace(case=case, rounds=3, slots=2, launches=2, fault=fault)
    compiled = D.compile(make_program(case, 3, 2), profile=D.NATIVE_PROFILE)
    reports = [{"rank": rank, "completed": True, "closed": True, "launches": []} for rank in range(2)]
    for rank in range(2):
        (tmp_path / f"rank{rank}").mkdir()
    for launch in range(2):
        a, b = make_inputs(3, launch)
        np.savez(tmp_path / f"launch{launch}_inputs.npz", original_a=a, original_b=b)
        if corrupt and launch == 0:
            a = a.copy()
            a[0, 0, 0, 0] = np.uint16(0x4080)
        if case == "gemm":

            def decode(value):
                return (value.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

            expected = np.swapaxes(decode(a) @ decode(b), 0, 1).astype(np.float32)
            outputs = [expected.copy(), expected.copy()]
        else:
            outputs = [a[1].copy(), a[0].copy()]
        for rank, actual in enumerate(outputs):
            np.save(tmp_path / f"rank{rank}/launch{launch}_output.npy", actual)
            plan = compiled.plan(epoch_base=launch * 3, epoch_end=(launch + 1) * 3)[rank]
            phases = [{"index": i, "op": instruction[0]} for i, instruction in enumerate(plan)]
            signals = expected_signals(3, 2, launch * 3).tolist()
            reports[rank]["launches"].append(
                {
                    "signals_passed": True,
                    "ready": signals,
                    "ack": signals,
                    "execution": {
                        "passed": True,
                        "completed": True,
                        "device_executed": True,
                        "kernel_executed": True,
                        "rank": rank,
                        "epoch_base": launch * 3,
                        "epoch_end": (launch + 1) * 3,
                        "phases_count": len(phases),
                        "phases": phases,
                    },
                }
            )
    return args, reports


@pytest.mark.parametrize("case", ["gemm", "put", "get"])
def test_global_oracle_accepts_correct_complete_native_outputs(tmp_path, case):
    args, reports = _results(tmp_path, case)
    result = validate_outputs(tmp_path, args, reports)
    assert result["passed"] and result["report_complete"]
    assert not result["expected_negative_detected"]


@pytest.mark.parametrize("case", ["gemm", "put", "get"])
def test_negative_requires_remote_result_mismatch_and_completed_execution(tmp_path, case):
    args, reports = _results(tmp_path, case, fault="corrupt_send", corrupt=True)
    result = validate_outputs(tmp_path, args, reports)
    assert not result["passed"] and result["expected_negative_detected"]
    assert result["accuracy"][0]["ranks"][1]["mismatches"] > 0
    assert result["fault_reference_passed"]
    assert result["accuracy"][1]["passed"]
    reports[0]["launches"][0]["execution"]["passed"] = False
    result = validate_outputs(tmp_path, args, reports)
    assert not result["report_complete"] and not result["expected_negative_detected"]


@pytest.mark.parametrize("damage", ["unwritten", "unrelated", "missing_phase", "bad_epoch", "missing_rank"])
def test_negative_rejects_incomplete_or_unrelated_failures(tmp_path, damage):
    args, reports = _results(tmp_path, "gemm", fault="corrupt_send", corrupt=True)
    path = tmp_path / "rank1/launch0_output.npy"
    if damage in ("unwritten", "unrelated"):
        actual = np.load(path)
        if damage == "unwritten":
            actual.fill(np.nan)
        else:
            actual.flat[-1] += 1
        np.save(path, actual)
    elif damage == "missing_phase":
        reports[1]["launches"][0]["execution"]["phases"].pop()
    elif damage == "bad_epoch":
        reports[1]["launches"][0]["execution"]["epoch_base"] = 7
    else:
        reports.pop()
    result = validate_outputs(tmp_path, args, reports)
    assert not result["passed"] and not result["expected_negative_detected"]


def test_requested_fault_cannot_pass_when_input_was_not_corrupted(tmp_path):
    args, reports = _results(tmp_path, "gemm", fault="corrupt_send")
    result = validate_outputs(tmp_path, args, reports)
    assert result["accuracy"][0]["passed"]
    assert not result["passed"] and not result["expected_negative_detected"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process group cleanup")
@pytest.mark.parametrize("leader_exits_first", [False, True])
def test_stop_workers_kills_term_ignoring_descendants(tmp_path, leader_exits_first):
    ready = tmp_path / "child.pid"
    child = (
        "import os,signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(120)"
        % str(ready)
    )
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(%d)" % (
        child,
        0 if leader_exits_first else 120,
    )
    process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "child did not start"
        pid = int(ready.read_text())
        if leader_exits_first:
            process.wait(timeout=5)
        stop_workers([process])
        deadline = time.monotonic() + 5
        path = Path(f"/proc/{pid}/stat")
        while path.exists() and path.read_text().split()[2] != "Z" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not path.exists() or path.read_text().split()[2] == "Z"
        assert process.poll() is not None
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_expected_signal_epochs_cover_slot_reuse_and_unused_slots():
    np.testing.assert_array_equal(expected_signals(3, 2, 6), [9, 8])
    np.testing.assert_array_equal(expected_signals(1, 2, 4), [5, 0])
    np.testing.assert_array_equal(expected_signals(3, 1, 3), [6])
