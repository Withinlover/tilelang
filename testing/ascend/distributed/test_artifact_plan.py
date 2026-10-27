"""Static contracts retain dynamic signal checks without walking compute IR."""

import json
import random
from types import SimpleNamespace

import pytest

import tilelang
import tilelang.distributed as D
from examples.distributed.example_allgather_gemm import allgather_gemm
from examples.distributed.example_fused_exchange import vector_exchange
from examples.distributed.example_p0_shapes import allgather_dense_gemm
from examples.distributed.run_native_distributed import exchange
from tilelang.ascend.distributed.artifact_plan import StaticLaunch, make_static_contract
from tilelang.ascend.distributed.native import NativeCompiledProgram
from tilelang.ascend.distributed.plan import SignalRef
from testing.ascend.distributed.test_native_plan import fake_headers


def outcome(plans, base, initial):
    try:
        return NativeCompiledProgram._check_progress(plans, base, initial_signals=initial)
    except ValueError:
        # Independent failures can change which cell is reported first. The
        # contract promises the same rejection, not an identical diagnostic.
        return None


@pytest.mark.parametrize("factory", [exchange, lambda r, s: exchange(r, s, get=True), vector_exchange, allgather_gemm])
@pytest.mark.parametrize("rounds,slots", [(1, 2), (7, 1)])
def test_signal_projection_matches_full_protocol_plan(factory, rounds, slots):
    kernel = D.compile(factory(rounds, slots), profile=D.P0B_PROFILE if factory is allgather_gemm else D.P0A_PROFILE)
    normal = kernel.plan()
    contract = make_static_contract(normal, proof=(0, 100), scalars=(), descriptions=kernel._describe_plans(normal))
    assert contract is not None
    for base in (0, 19):
        projected, _ = contract.signal_plans(base, base + rounds)
        full = kernel.plan(epoch_base=base, epoch_end=base + rounds)
        for initial in (None, {}, {(0, "Ready", 0): 100}, {(1, "Ack", 0): 0}):
            assert outcome(full, base, initial) == outcome(projected, base, initial)


def test_signal_projection_matches_random_single_writer_schedules():
    rng = random.Random(80193)
    accepted = rejected = 0
    for _ in range(500):
        plans = {0: [], 1: []}
        for rank, instructions in plans.items():
            for _ in range(rng.randrange(2, 18)):
                key, cell, offset = rng.choice(("Ready", "Ack")), rng.randrange(2), rng.randrange(1, 6)
                kind = rng.choice(("signal", "put_signal", "wait_signal"))
                ref = SignalRef(rank if kind == "wait_signal" else 1 - rank, key, cell)
                instructions.extend(("gemm",) for _ in range(rng.randrange(7)))
                instructions.append((kind, None, None, None, ref, offset) if kind == "put_signal" else (kind, ref, offset, ()))
        contract = make_static_contract(plans, proof=(0, 100), scalars=(), descriptions={})
        assert contract is not None
        base = 11
        projected, _ = contract.signal_plans(base, base + 5)
        full = {
            rank: [
                (*item[:5], item[5] + base)
                if item[0] == "put_signal"
                else (item[0], item[1], item[2] + base, ())
                if item[0] in ("signal", "wait_signal")
                else item
                for item in instructions
            ]
            for rank, instructions in plans.items()
        }
        initial = {(rank, key, cell): rng.randrange(17) for rank in (0, 1) for key in ("Ready", "Ack") for cell in (0, 1)}
        expected = outcome(full, base, initial)
        assert outcome(projected, base, initial) == expected
        accepted += expected is not None
        rejected += expected is None
    assert accepted and rejected


def test_contract_keeps_compute_out_of_handshake_and_freezes_phase_reports():
    compact = {
        rank: [("signal", SignalRef(1 - rank, "Ready", 0), 1, ()), ("wait_signal", SignalRef(rank, "Ready", 0), 1, ())] for rank in (0, 1)
    }
    expanded = {rank: [("gemm",)] * 10000 + plan for rank, plan in compact.items()}
    contracts = [make_static_contract(plan, proof=(0, 100), scalars=(), descriptions={}) for plan in (compact, expanded)]
    assert contracts[0].signals == contracts[1].signals
    metadata = [StaticLaunch(contract, SimpleNamespace(artifact_id="fixed"), 0, 1, {}).metadata() for contract in contracts]
    assert len(json.dumps(metadata[1])) - len(json.dumps(metadata[0])) < 16
    phases = contracts[1].phases[0]
    assert json.loads(json.dumps(phases)) == [{"index": i, "op": item[0]} for i, item in enumerate(expanded[0])]
    with pytest.raises(TypeError, match="read-only"):
        phases.append({"index": 10002, "op": "quiet"})
    with pytest.raises(TypeError, match="read-only"):
        phases[0]["op"] = "signal"


@pytest.mark.parametrize("proof,ref", [(None, SignalRef(1, "Ready", 0)), ((0, 10), SignalRef(0, "Ready", 0))])
def test_projection_falls_back_without_epoch_proof_or_remote_single_writer(proof, ref):
    assert make_static_contract({0: [("signal", ref, 1, ())], 1: []}, proof=proof, scalars=(), descriptions={}) is None


def test_large_device_loop_contract_is_constant_size_and_public_plan_stays_full(monkeypatch, tmp_path):
    root = fake_headers(tmp_path)
    kernel = D.compile(allgather_dense_gemm(), profile=D.P0B_PROFILE, shmem_root=root, execution_policy="artifact")
    monkeypatch.setattr(tilelang, "compile", lambda *args, **kwargs: lambda *values: None)
    kernel.build(shmem_root=root)
    public = kernel.plan(epoch_base=3)
    assert sum(item[0] == "gemm" for item in public[0]) == 2
    launch = kernel._launch_plan(epoch_base=3)
    assert isinstance(launch, StaticLaunch)
    assert len(launch.contract.signals[0]) == 4
    assert len(json.dumps(kernel._describe_plans(launch))) < 1000
    assert kernel._prepare_executable(shmem_root=root, plans=launch)


@pytest.mark.parametrize("fallback", ["unproven", "overflow", "new_scalar", "evicted_kernel"])
def test_static_launch_fallback_retains_full_planning(monkeypatch, tmp_path, fallback):
    from tvm import tirx
    from tilelang.distributed.abi import MAX_EPOCH

    source = exchange(1, 1)
    epoch = next(param for param in source.params if param.name == "epoch0")
    if fallback == "unproven":
        source = source.with_body(tirx.IfThenElse(epoch >= 2, source.body, source.body))
    elif fallback == "overflow":
        source = source.with_body(tirx.stmt_functor.substitute(source.body, {epoch: (epoch + 10) - 10}))
    elif fallback == "new_scalar":
        source = tirx.PrimFunc([*source.params, tirx.Var("choice", "int32")], source.body, buffer_map=source.buffer_map, attrs=source.attrs)
    root = fake_headers(tmp_path)
    kernel = D.compile(source, profile=D.P0A_PROFILE, shmem_root=root, execution_policy="artifact")
    monkeypatch.setattr(tilelang, "compile", lambda *args, **kwargs: lambda *values: None)
    kernel.build(shmem_root=root, scalars={"choice": 0} if fallback == "new_scalar" else None)
    if fallback == "evicted_kernel":
        kernel._kernels.clear()
    calls, original = [], kernel.plan

    def counted(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(kernel, "plan", counted)
    if fallback == "overflow":
        with pytest.raises(ValueError, match="integer overflow"):
            kernel._launch_plan(epoch_base=MAX_EPOCH - 2, epoch_end=MAX_EPOCH)
    else:
        result = kernel._launch_plan(epoch_base=2, scalars={"choice": 1} if fallback == "new_scalar" else None)
        assert isinstance(result, dict)
    assert len(calls) == 1
