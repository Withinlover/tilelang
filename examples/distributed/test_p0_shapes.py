"""Dense shape benchmark regression: layout, true K accumulation and evidence."""

import json
import re
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import tilelang.testing

from examples.distributed.benchmark_p0_shapes import compact_case, parse_shapes, save_sweep
from examples.distributed.example_p0_shapes import (
    allgather_dense_gemm,
    bulk_exchange,
    check_dense_output,
    local_dense_gemm,
    make_dense_inputs,
    pack_inputs,
    pack_local_inputs,
    reference_output,
    tile_counts,
    unpack_output,
    workload,
)

requires_ascend_backend = pytest.mark.skipif(
    not tilelang.testing.ascend_backend_compiled(), reason="Ascend compiler backend is not built; no NPU is required"
)


@pytest.mark.parametrize("shape", [(64, 128, 128), (128, 64, 64), (128, 128, 256)])
def test_packed_layout_matches_independent_dense_matmul(shape):
    a, b = make_dense_inputs(*shape, seed=19)
    pa, pb = pack_inputs(a, b)
    for rank in range(2):
        local_a, local_b = pack_local_inputs(a[rank], b)
        np.testing.assert_array_equal(local_a, pa[rank])
        np.testing.assert_array_equal(local_b, pb)
    af = (pa.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    bf = (pb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    mt, nt, kt = tile_counts(*shape)
    result = np.zeros((2, mt, nt, 64, 64), dtype=np.float32)
    for owner in range(2):
        for row in range(mt):
            for col in range(nt):
                result[owner, row, col] = sum(af[owner, row, inner] @ bf[col, inner] for inner in range(kt))
    check = check_dense_output(unpack_output(result), reference_output(a, b))
    assert check["passed"] and check["elements"] == 2 * shape[0] * shape[1]
    assert not a.flags.writeable and not b.flags.writeable


def test_reference_detects_dropped_late_k_tile_and_nonfinite():
    a = np.zeros((2, 64, 128), dtype=np.uint16)
    b = np.zeros((128, 64), dtype=np.uint16)
    a[:, :, 127] = np.float32(2).view(np.uint32) >> 16
    b[127, :] = np.float32(3).view(np.uint32) >> 16
    reference = reference_output(a, b)
    assert np.array_equal(reference, np.full((128, 64), 6.0))
    wrong = np.zeros((128, 64), dtype=np.float32)
    check = check_dense_output(wrong, reference)
    assert not check["passed"] and check["mismatches"] == wrong.size
    wrong[0, 0] = np.nan
    assert check_dense_output(wrong, reference)["nonfinite"] == 1


@pytest.mark.parametrize("shape", [(0, 64, 64), (64, 65, 64), (1024, 1024, 1088), (True, 64, 64)])
def test_unsupported_dense_shape_is_diagnostic(shape):
    with pytest.raises(ValueError, match="multiple of 64"):
        allgather_dense_gemm(*shape)


def test_shape_sweep_and_true_work_counts():
    assert parse_shapes("128,64x128x256,1024") == [(128, 128, 128), (64, 128, 256), (1024, 1024, 1024)]
    for text in ("128,128", "128x64", "1025", "64x128x0"):
        with pytest.raises(ValueError):
            parse_shapes(text)
    work = workload(1024, 1024, 1024)
    assert work["flops_per_rank"] == 4 * 1024**3
    assert work["logical_peer_payload_bytes_per_rank"] == 2 * 1024**2
    assert work["output_shape_per_rank"] == [2048, 1024]
    assert work["k_tiles"] == 16


def test_worker_timeout_argument_remains_native_integer(monkeypatch):
    from examples.distributed import benchmark_p0_shapes as benchmark

    observed = {}

    def fake_worker(args):
        observed["timeout"] = args.operation_timeout
        return 0

    monkeypatch.setattr(benchmark, "worker", fake_worker)
    assert benchmark.main(["--worker", "--operation-timeout", "180"]) == 0
    assert type(observed["timeout"]) is int and observed["timeout"] == 180


def test_torchrun_forwards_unambiguous_matrix_dimensions(monkeypatch):
    from torch.distributed.run import parse_args
    from examples.distributed import benchmark_p0_shapes as benchmark

    forwarded = ["--worker", "--stage", "device", "--matrix-m", "128", "--matrix-n", "256", "--matrix-k", "1024"]
    parsed = parse_args(["--standalone", "--nproc_per_node=2", "-m", "examples.distributed.benchmark_p0_shapes", *forwarded])
    assert parsed.training_script_args == forwarded
    observed = {}

    def fake_worker(args):
        observed["shape"] = args.m, args.n, args.k
        return 0

    monkeypatch.setattr(benchmark, "worker", fake_worker)
    assert benchmark.main(parsed.training_script_args) == 0
    assert observed["shape"] == (128, 256, 1024)


@pytest.mark.parametrize("factory", [allgather_dense_gemm, bulk_exchange, local_dense_gemm])
@requires_ascend_backend
def test_sources_preserve_large_tensor_shapes(factory):
    program = factory(1024, 1024, 1024)
    assert any(tuple(int(value) for value in buffer.shape) == (16, 16, 64, 64) for buffer in program.buffer_map.values())


@requires_ascend_backend
def test_p0b_dense_lower_preserves_device_loops_and_one_mixed_kernel(tmp_path):
    import tilelang.distributed as D

    compiled = D.compile(allgather_dense_gemm(128, 128, 128), profile=D.P0B_PROFILE)
    plans = compiled.plan()
    for rank in range(2):
        assert sum(instruction[0] == "gemm" for instruction in plans[rank]) == 2
    result = compiled.lower(output_dir=tmp_path)
    assert result["device_executed"] is False
    for artifact in result["artifacts"]:
        source = (tmp_path / artifact["source"]).read_text()
        assert source.count("__global__") == 1 and "__mix__" in source
        assert source.count("asc_mmad(") == 2
        assert source.count("for (") == 6
        assert "inner == 0" in source
        cube, vector = source.split("if ASC_IS_AIV", 1)
        cube = cube.split("if ASC_IS_AIC", 1)[1]
        assert cube.index("(&(Send[") < cube.index("asc_sync_intra_wait(PIPE_MTE2") < cube.index("(&(Recv[")
        released = re.search(r"asc_sync_intra_arrive\(PIPE_MTE2, (\d+)\)", cube)
        acquired = re.search(r"asc_sync_intra_wait\(PIPE_MTE2, (\d+)\)", cube)
        assert released and acquired
        assert cube.index("(&(Recv[") < released.start() < cube.rindex("asc_mmad(")
        guard = cube[cube.rfind("if (", 0, released.start()) : released.start()]
        assert all(re.search(rf"{name}(?:_\d+)? == 1", guard) for name in ("row", "col", "inner"))
        assert vector.index("distributed_task<4>") < vector.index(f"asc_sync_intra_arrive(PIPE_S, {acquired[1]})")
        assert vector.index(f"asc_sync_intra_wait(PIPE_S, {released[1]})") < vector.index("distributed_task<3>")


def test_compact_failed_build_is_not_device_evidence(tmp_path):
    entry = {
        "shape": [1024, 1024, 1024],
        "mode": "p0b",
        "stage": "compile",
        "status": "failed",
        "exit_code": 1,
        "timed_out": False,
        "device_executed": False,
        "rank_reports": [{"error_type": "ValueError", "error": "private host /home/secret/path"}],
    }
    compact = compact_case(entry)
    assert compact["execution"]["median_ms"] is None
    assert compact["error_types"] == ["ValueError"]
    saved = save_sweep(tmp_path, [entry], "a" * 40)
    assert saved["passed"] is False and saved["full_device_acceptance"] == "not_assessed"
    text = (tmp_path / "summary.json").read_text()
    assert "secret" not in text and "private" not in text
    assert json.loads(text)["cases"][0]["device_executed"] is False


def test_compact_size_does_not_grow_with_iteration_count(tmp_path):
    entry = {
        "shape": [128, 128, 128],
        "mode": "hccl",
        "stage": "device",
        "status": "passed",
        "exit_code": 0,
        "timed_out": False,
        "device_executed": True,
        "summary": {"ranks": [{"rank": 0, "timing_samples": [{"arbitrary": "secret"}] * 2000}]},
    }
    save_sweep(tmp_path, [entry], None)
    assert len((tmp_path / "summary.json").read_bytes()) < 5000
    assert "timing_samples" not in (tmp_path / "summary.json").read_text()


@pytest.mark.parametrize("mode", ["hccl", "p0a", "p0b"])
def test_shape_worker_keeps_host_timing_and_separate_stream_intervals(tmp_path, monkeypatch, mode):
    import torch
    import torch.distributed as dist
    import tilelang.distributed as D
    from examples.distributed import benchmark_p0_shapes as benchmark

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    state = {"calls": 0, "events_prepared": False}

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.0

    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda _: None, synchronize=lambda _: None, Event=Event), raising=False)
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: original_empty(*args, **{**kwargs, "device": "cpu"}))
    original_inputs = benchmark.make_dense_inputs

    def inputs(*args):
        a, b = original_inputs(*args)
        state["a"], state["b"] = pack_inputs(a, b)
        state["expected"] = reference_output(a, b)
        return a, b

    monkeypatch.setattr(benchmark, "make_dense_inputs", inputs)

    def local(a0, a1, b, output):
        state["calls"] += 1
        packed = state["expected"].astype(np.float32).reshape(2, 2, 64, 2, 64).transpose(0, 1, 3, 2, 4).copy()
        output.copy_(torch.from_numpy(packed))

    objects = [
        {"key": key, "shape": shape, "dtype": dtype, "kind": "symmetric", "remote_access": "read"}
        for key, shape, dtype in (
            ("Send", (2, 2, 64, 64), "bfloat16"),
            ("Recv", (2, 2, 64, 64), "bfloat16"),
            ("B", (2, 2, 64, 64), "bfloat16"),
            ("Output", (2, 2, 2, 64, 64), "float32"),
        )
    ]
    compiled = SimpleNamespace(profile="test", manifest={"objects": objects})
    monkeypatch.setattr(benchmark, "compile_workload", lambda args: (None if mode == "p0b" else local, compiled, 0.5))

    def copies(value):
        return [{**deepcopy(value), "rank": rank} for rank in range(2)]

    if mode == "hccl":
        monkeypatch.setattr(dist, "init_process_group", lambda *args, **kwargs: None)
        monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
        monkeypatch.setattr(dist, "barrier", lambda: None)
        monkeypatch.setattr(
            dist,
            "all_gather",
            lambda buffers, source: [
                buffer.copy_(torch.from_numpy(state["a"][rank]).view(torch.bfloat16)) for rank, buffer in enumerate(buffers)
            ],
        )
        monkeypatch.setattr(dist, "all_gather_object", lambda target, value: target.__setitem__(slice(None), copies(value)))
    else:

        def prepare_events():
            state["events_prepared"] = True

        session = SimpleNamespace(
            preflight=lambda value: None,
            control=SimpleNamespace(all_gather=copies),
            prepare_device_timing=prepare_events,
            copy_out=lambda value: state["expected"].astype(np.float32).reshape(2, 2, 64, 2, 64).transpose(0, 1, 3, 2, 4).copy(),
        )
        monkeypatch.setattr(D, "init", lambda **kwargs: session)
        monkeypatch.setattr(D, "alloc_symmetric", lambda session, **kwargs: SimpleNamespace(view=lambda **kwargs: None))
        monkeypatch.setattr(D, "channel", lambda *args, **kwargs: None)
        monkeypatch.setattr(D, "prepare_launch", lambda *args, **kwargs: None)
        monkeypatch.setattr(D, "copy_in", lambda *args, **kwargs: None)

        def launch(*args, **kwargs):
            assert kwargs["measure_device"] is True and state["events_prepared"]
            return {
                "physical_kernel_count": 1,
                "execution_policy": "artifact",
                "source_validation": "identity",
                "launch_validation": "static_contract",
                "timing_basis": "host_wall",
                "timing_seconds": {"plan_validation": 0.001},
                "device_stream_seconds": 0.003,
                "device_timing_basis": "acl_event_stream_interval_including_submission_gaps",
            }

        monkeypatch.setattr(D, "launch", launch)
        monkeypatch.setattr(D, "complete", lambda result: result)
        monkeypatch.setattr(D, "close", lambda session: None)
    args = SimpleNamespace(
        output_dir=tmp_path,
        m=128,
        n=128,
        k=128,
        mode=mode,
        stage="device",
        execution_policy="artifact",
        devices=[0, 1],
        operation_timeout=60,
        bootstrap_port=29720,
        control_port=29721,
        shmem_root=None,
        shmem_library=None,
        measure_device=True,
        warmup=1,
        iterations=2,
    )
    assert benchmark.worker(args) == 0
    result = json.loads((tmp_path / "detail.json").read_text())
    assert result["passed"] and result["execution"]["count"] == 2
    assert state["calls"] == (0 if mode == "p0b" else 3)
    intervals = result["device_intervals"]
    assert (
        set(intervals) == {"collective_and_local_gemm", "collective", "local_gemm"}
        if mode == "hccl"
        else set(intervals) == ({"communication", "local_gemm"} if mode == "p0a" else {"fused"})
    )
    for sample in result["ranks"][0]["timing_samples"]:
        assert sample["end_to_end_seconds"] >= sample["execution_seconds"] > 0
        assert sample["stages_seconds"]["pack"] >= 0 and sample["stages_seconds"]["unpack"] >= 0
