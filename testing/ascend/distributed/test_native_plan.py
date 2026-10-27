"""Hardware-free native phase planning, source generation and launch ABI checks."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.language.distributed as TD
from tilelang.ascend.distributed.native import NativeCompiledProgram, _segment, _segment_key, describe_instruction
from tilelang.ascend.distributed.plan import SignalRef, plan_program
from examples.distributed.example_allgather_gemm import allgather_gemm


G = TD.GroupSpec("world", 2)
CH = TD.ChannelSpec("comm", G)


def transfer(*, offset=0, count=8, get=False):
    @T.prim_func
    def program(
        Send: TD.SymBuffer((16,), "float32", group=G, remote_access="read"),
        Recv: TD.SymBuffer((16,), "float32", group=G, remote_access="write"),
        Ready: TD.SignalBuffer((1,), group=G),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            peer = 1 - TD.rank(G)
            if get:
                TD.signal(TD.remote(Ready, peer)[0], epoch0 + 1, channel=CH, release_regions=[Send[:]])
                TD.wait_signal(Ready[0], epoch0 + 1, channel=CH, covers=[TD.remote(Send, peer)[:]])
                done = TD.copy_async(TD.remote(Send, peer)[offset : offset + count], Recv[offset : offset + count], channel=CH)
                TD.wait(done, completion="local")
            else:
                done = TD.put_signal(
                    Send[offset : offset + count],
                    TD.remote(Recv, peer)[offset : offset + count],
                    signal=TD.remote(Ready, peer)[0],
                    value=epoch0 + 1,
                    channel=CH,
                )
                TD.wait(done, completion="remote")
                TD.wait_signal(Ready[0], epoch0 + 1, channel=CH, covers=[Recv[offset : offset + count]])
            TD.quiet(channel=CH)

    return program


def fake_headers(tmp_path):
    root = tmp_path / "shmem"
    (root / "include").mkdir(parents=True)
    (root / "src/device/gm2gm/engine").mkdir(parents=True)
    (root / "include/shmem.h").write_text("// mock build metadata only\n")
    (root / "src/device/gm2gm/engine/shmem_device_mte.hpp").write_text("// mock MTE ABI\n")
    return root


def test_same_source_plans_native_rank_slots_ack_and_output_ownership(monkeypatch):
    monkeypatch.delenv("CANNSIM_ROOT", raising=False)
    compiled = NativeCompiledProgram(allgather_gemm(rounds=3, slots=2))
    plans = compiled.plan(epoch_base=7, epoch_end=10)
    for rank, plan in plans.items():
        puts = [item for item in plan if item[0] == "put_signal"]
        assert [item[5] for item in puts] == [8, 9, 10]
        assert [item[2].offset for item in puts] == [0, 4096, 8192]
        assert [item[3].offset for item in puts] == [0, 4096, 0]
        assert all(item[2].rank == rank and item[3].rank == 1 - rank for item in puts)
        reuse = plan.index(puts[2])
        assert plan[reuse - 1] == ("wait_signal", SignalRef(rank, "Ack", 0), 8, ())
        gemms = [item for item in plan if item[0] == "gemm"]
        assert len(gemms) == 6
        assert [item[2].offset for item in gemms[:2]] == [rank * 4096, (1 - rank) * 4096]
        for gemm in gemms:
            local = _segment(gemm)
            assert len(local.params) == 3
            assert "tl.distributed" not in local.script()
            assert "__comm_descriptor" not in local.script()
    assert compiled.manifest["descriptor"]["device_argument"] is False
    assert compiled.source_manifest["descriptor"]["encoding"] == "cpu-model-v1"


def test_native_and_shared_planner_use_identical_operations():
    source = allgather_gemm(rounds=1)
    compiled = NativeCompiledProgram(source)
    for rank in (0, 1):
        shared, _ = plan_program(source, rank=rank, scalars={"epoch0": 0}, world_size=2)
        native = compiled.plan()[rank]
        assert [describe_instruction(item) for item in shared] == [describe_instruction(item) for item in native]


@pytest.mark.parametrize(("offset", "count"), [(1, 8), (0, 7)])
def test_native_mte_rejects_unaligned_regions_before_any_runtime(offset, count):
    compiled = NativeCompiledProgram(transfer(offset=offset, count=count))
    with pytest.raises(ValueError, match="32-byte"):
        compiled.plan()


@pytest.mark.parametrize("get", [False, True])
def test_native_plan_retains_put_and_get_directions(get):
    compiled = NativeCompiledProgram(transfer(get=get))
    plan = compiled.plan()[0]
    copy = next(item for item in plan if item[0] in ("copy_async", "put_signal"))
    assert compiled._transport_spec(copy, 0) == (("get" if get else "put_signal"), 32)


def test_epoch_and_compiled_source_identity_are_fail_closed():
    compiled = NativeCompiledProgram(allgather_gemm(rounds=3))
    with pytest.raises(ValueError, match="published epoch"):
        compiled.plan(epoch_end=2)
    with pytest.raises(ValueError, match="epoch0"):
        compiled.plan(scalars={"epoch0": 4}, epoch_base=0)
    compiled.manifest["objects"][0]["shape"][0] = 100
    with pytest.raises(ValueError, match="modified"):
        compiled.plan()


def test_native_header_content_enters_manifest_and_rejects_post_compile_change(tmp_path):
    root = fake_headers(tmp_path)
    first = NativeCompiledProgram(transfer(), shmem_root=root)
    (root / "include/shmem.h").write_text("// revised ABI\n")
    second = NativeCompiledProgram(transfer(), shmem_root=root)
    assert first.header_fingerprint != second.header_fingerprint
    assert first.manifest["transport_header_sha256"] != second.manifest["transport_header_sha256"]
    with pytest.raises(ValueError, match="changed after native compile"):
        first.build(shmem_root=root)


def test_native_source_only_lowering_needs_no_helper_or_shmem(tmp_path, monkeypatch):
    monkeypatch.delenv("CANNSIM_ROOT", raising=False)
    monkeypatch.delenv("SHMEM_ROOT", raising=False)
    compiled = NativeCompiledProgram(allgather_gemm(rounds=1), output_dir=tmp_path)
    report = compiled.lower()
    assert not report["device_executed"]
    assert {item["kind"] for item in report["artifacts"]} == {"put_signal", "wait_signal", "signal", "gemm"}
    for artifact in report["artifacts"]:
        source = (tmp_path / artifact["source"]).read_text()
        assert "tl.distributed" not in source and "__comm_descriptor" not in source
        assert "__global__ __cube__" in source if artifact["kind"] == "gemm" else "__global__ __vector__" in source
    with pytest.raises(ValueError, match="transport is not certified"):
        tilelang.lower(compiled.prim_func, target="ascend")


@dataclass
class View:
    key: str
    shape: tuple
    dtype: str
    byte_offset: int


class Buffer:
    def __init__(self, spec):
        self.key, self.dtype = spec["key"], spec["dtype"]

    def view(self, *, shape, dtype=None, byte_offset=0):
        return View(self.key, shape, dtype or self.dtype, byte_offset)


def test_native_run_binds_real_phase_abis_without_passing_cpu_descriptor(tmp_path, monkeypatch):
    root = fake_headers(tmp_path)
    compiled = NativeCompiledProgram(allgather_gemm(rounds=2, slots=2), shmem_root=root, output_dir=tmp_path)
    calls, preflights = [], []
    plans = compiled.plan()

    def build(**kwargs):
        for rank, instructions in kwargs["plans"].items():
            for instruction in instructions:
                spec = compiled._transport_spec(instruction, rank)
                if spec is not None:
                    compiled._kernels[("transport", *spec, compiled.header_fingerprint, "native")] = spec[0]
                elif instruction[0] == "gemm":
                    compiled._kernels[("gemm", _segment_key(instruction), "native")] = "gemm"

    monkeypatch.setattr(compiled, "build", build)
    session = SimpleNamespace(
        profile=compiled.profile,
        world_size=2,
        rank=0,
        shmem_root=root,
        abi_fingerprint="mock-ABI",
        objects={spec["key"]: Buffer(spec) for spec in compiled.manifest["objects"]},
        preflight=preflights.append,
        synchronize=lambda: calls.append(("sync", [])),
        launch_kernel=lambda kernel, *args: calls.append((kernel, args)),
    )
    ticket = SimpleNamespace(manifest=compiled.manifest, kernel=compiled, base=0, end=2, channel=SimpleNamespace(peers={0: (1,), 1: (0,)}))
    report = compiled.run(session=session, ticket=ticket)
    assert report["passed"] and report["completed"]
    assert len(preflights) == 2
    first = calls[0]
    assert first[0] == "put_signal"
    assert first[1] == (View("Recv", (8192,), "uint8", 0), View("Send", (8192,), "uint8", 0), 1, View("Ready", (16,), "uint32", 0), 1)
    second_put = [call for call in calls if call[0] == "put_signal"][1]
    assert second_put[1][3] == View("Ready", (16,), "uint32", 64)
    assert second_put[1][4] == 2
    expected = [item[0] for item in plans[0] if item[0] not in ("wait", "quiet", "fence")]
    assert [kind for kind, _ in calls] == expected + ["sync"]
    assert not any(isinstance(arg, View) and arg.key == "__comm_descriptor" for _, args in calls for arg in args)


@pytest.mark.parametrize("publish_after_wait", [False, True])
def test_definite_wait_cycle_or_absent_producer_is_rejected(publish_after_wait):
    @T.prim_func
    def stuck(Ready: TD.SignalBuffer((1,), group=G), epoch0: T.int32):
        with T.Kernel(1):
            TD.wait_signal(Ready[0], epoch0 + 1, channel=CH, covers=[])
            if publish_after_wait:
                TD.signal(TD.remote(Ready, 1 - TD.rank(G))[0], epoch0 + 1, channel=CH)
            TD.quiet(channel=CH)

    with pytest.raises(ValueError, match="definite signal wait deadlock"):
        NativeCompiledProgram(stuck).plan()


@pytest.mark.parametrize("get", [False, True])
def test_native_exchange_source_against_protocol_model_with_slot_reuse(tmp_path, get):
    """The exact hardware runner source also satisfies the CPU lease checker."""
    import os
    from pathlib import Path

    import numpy as np
    import tilelang.distributed as D
    from examples.distributed.example_allgather_gemm import make_inputs
    from examples.distributed.run_native_distributed import exchange

    helper = os.environ.get("CANNSIM_ROOT")
    if helper is None or not (Path(helper) / "cosim/protocol.py").is_file():
        pytest.skip("set CANNSIM_ROOT to enable protocol cross-validation of native examples")
    compiled = D.compile(exchange(3, 2, get=get), mode="protocol", helper_root=helper, output_dir=tmp_path)
    session = D.init()
    for spec in compiled.manifest["objects"]:
        common = dict(key=spec["key"], shape=spec["shape"], dtype=spec["dtype"])
        if spec["kind"] == "signal":
            D.alloc_signals(session, **common)
        else:
            D.alloc_symmetric(session, remote_access=spec["remote_access"], **common)
    channel = D.channel(session, key="comm")
    for launch in range(2):
        original, _ = make_inputs(3, seed=launch)
        ticket = D.prepare_launch(session, compiled, channel=channel, epochs=3, protocol="native-exchange-source")
        D.copy_in(session.objects["Input"], {rank: original[rank] for rank in range(2)}, ticket=ticket)
        report = D.complete(D.launch(compiled, ticket=ticket, seed=launch))
        assert report["passed"] and report["protocol"]["passed"]
        assert not report["protocol"]["open_leases"]
        assert not report["protocol"]["pending_events"]
        assert report["epoch_base"] == launch * 3
        for rank in range(2):
            np.testing.assert_array_equal(session.objects["Output"].arrays[rank], original[1 - rank])
    D.close(session)


def test_progress_check_accepts_rank_specific_producer_and_prior_epoch_wait():
    @T.prim_func
    def handshake(Ready: TD.SignalBuffer((1,), group=G), epoch0: T.int32):
        with T.Kernel(1):
            rank = TD.rank(G)
            if rank == 0:
                TD.signal(TD.remote(Ready, 1)[0], epoch0 + 1, channel=CH)
                TD.wait_signal(Ready[0], epoch0 + 1, channel=CH, covers=[])
            else:
                TD.wait_signal(Ready[0], epoch0 + 1, channel=CH, covers=[])
                TD.signal(TD.remote(Ready, 0)[0], epoch0 + 1, channel=CH)
            TD.quiet(channel=CH)

    assert set(NativeCompiledProgram(handshake).plan()) == {0, 1}

    @T.prim_func
    def old_epoch(Ready: TD.SignalBuffer((1,), group=G), epoch0: T.int32):
        with T.Kernel(1):
            TD.wait_signal(Ready[0], epoch0, channel=CH, covers=[])
            TD.quiet(channel=CH)

    # An earlier launch may have supplied this value. Lack of a producer in
    # the current plan alone is insufficient to prove this wait deadlocks.
    assert set(NativeCompiledProgram(old_epoch).plan(epoch_base=1)) == {0, 1}


@pytest.mark.parametrize("destination", [8, 16])
def test_native_local_copy_requires_disjoint_storage(destination):
    @T.prim_func
    def overlap(Data: TD.SymBuffer((32,), "float32", group=G, remote_access="read_write")):
        with T.Kernel(1):
            T.copy(Data[0:16], Data[destination : destination + 16])
            TD.quiet(channel=CH)

    compiled = NativeCompiledProgram(overlap)
    if destination == 8:
        with pytest.raises(ValueError, match="disjoint source and destination"):
            compiled.plan()
    else:
        assert compiled.plan()[0][0][0] == "local_copy"


def test_shmem_fingerprint_traverses_only_header_subtrees_and_preserves_digest(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from tilelang.ascend.distributed.transport import shmem_include_flags

    root = fake_headers(tmp_path)
    (root / "src/device_simt").mkdir()
    (root / "src/device_simt/extra.hpp").write_text("simt header")
    (root / "build").mkdir()
    (root / "build/ignored.hpp").write_text("not a selected include")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".h", ".hpp"} and any(path.is_relative_to(root / tree) for tree in ("include", "src")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    expected = "-DTL_SHMEM_HEADER_DIGEST_" + digest.hexdigest() + "=1"
    original = Path.rglob

    def restricted(path, pattern):
        assert path != root, "hot path must not enumerate build/install/repository objects"
        return original(path, pattern)

    monkeypatch.setattr(Path, "rglob", restricted)
    assert shmem_include_flags(root)[-1] == expected
    (root / "src/device_simt/extra.hpp").write_text("modified header")
    assert shmem_include_flags(root)[-1] != expected
    (root / "src/device_simt/extra.hpp").unlink()
    assert shmem_include_flags(root)[-1] != expected
