"""Full DSL-to-protocol checks; CANN computation is tested separately."""

import json
from pathlib import Path

import pytest
import numpy as np

import tilelang.distributed as D
import tilelang.testing

from example_allgather_gemm import allgather_gemm, default_helper_root, global_accuracy, make_inputs, run


@pytest.fixture(scope="module")
def helper():
    # Protocol mode still constructs Ascend copy/GEMM intrinsics. It needs
    # their native registrations, but no physical NPU or CANN installation.
    if not tilelang.testing.ascend_backend_compiled():
        pytest.skip("Ascend backend is not built; configure with USE_ASCEND=ON")
    root = default_helper_root()
    if root is None or not (root / "cosim/protocol.py").is_file():
        pytest.skip("extended cannsim-helper unavailable; set CANNSIM_ROOT")
    return root


@pytest.mark.parametrize(("rounds", "slots", "seed"), [(1, 2, 0), (3, 2, 1), (5, 1, 9)])
def test_dsl_allgather_gemm_protocol(helper, tmp_path, rounds, slots, seed):
    report = run(helper_root=helper, output_dir=tmp_path, rounds=rounds, slots=slots, seed=seed)
    assert report["passed"] and report["global_accuracy"]["mismatches"] == 0
    assert report["protocol"]["passed"] and not report["kernel_executed"]
    directory = Path(report["run_dir"])
    assert "tl.distributed.put_signal" in (directory / "program.tir").read_text()
    assert json.loads((directory / "manifest.json").read_text())["passed"]


@pytest.mark.parametrize("fault", ["early_ready", "drop_signal", "drop_ack", "corrupt_payload"])
def test_dsl_protocol_faults_fail_global_acceptance(helper, tmp_path, fault):
    report = run(helper_root=helper, output_dir=tmp_path, rounds=3, slots=2, fault=fault)
    assert not report["passed"]
    assert report["protocol"]["fault_injected"]
    if fault == "corrupt_payload":
        assert report["adapter_passed"] and report["protocol"]["passed"]
        assert report["global_accuracy"]["mismatches"] > 0
    else:
        assert not report["protocol"]["passed"]


def test_dsl_second_launch_uses_new_epoch_without_resetting_signals(helper, tmp_path):
    kernel = D.compile(allgather_gemm(3, 2), mode="protocol", helper_root=helper, output_dir=tmp_path)
    session = D.init()
    objects = {}
    for spec in kernel.manifest["objects"]:
        common = dict(key=spec["key"], shape=spec["shape"], dtype=spec["dtype"])
        objects[spec["key"]] = (
            D.alloc_signals(session, **common)
            if spec["kind"] == "signal"
            else D.alloc_symmetric(session, remote_access=spec["remote_access"], **common)
        )
    channel = D.channel(session, key="comm")
    for launch_id in range(2):
        ticket = D.prepare_launch(session, kernel, channel=channel, epochs=3, protocol="allgather-gemm-v1")
        assert ticket.base == 3 * launch_id
        a, b = make_inputs(3, launch_id)
        D.copy_in(objects["Send"], {rank: a[rank] for rank in range(2)}, ticket=ticket)
        D.copy_in(objects["B"], {rank: b for rank in range(2)}, ticket=ticket)
        result = D.complete(D.launch(kernel, ticket=ticket))
        assert result["passed"]
        actual = np.stack([objects["Output"].arrays[rank] for rank in range(2)])
        assert global_accuracy(actual, a, b)["passed"]
        for key in ("Ready", "Ack"):
            for rank in range(2):
                assert list(objects[key].arrays[rank]) == [ticket.base + 3, ticket.base + 2]
    D.close(session)
