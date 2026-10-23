"""Execute canonical TD TIR against the optional local protocol transport."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.language.distributed as TD
import tilelang.distributed as D
from tilelang import tvm
from tvm import tirx


G = TD.GroupSpec("world", 2)
CH = TD.ChannelSpec("comm", G)


def exchange(*, get=False, partial_overlap=False, signal_offset=1, wait_offset=0):
    @T.prim_func
    def main(
        Send: TD.SymBuffer((8,), "float32", group=G, remote_access="read"),
        Recv: TD.SymBuffer((8,), "float32", group=G, remote_access="write"),
        Output: TD.SymBuffer((8,), "float32", group=G, remote_access="write"),
        Ready: TD.SignalBuffer((1,), group=G),
        Ack: TD.SignalBuffer((1,), group=G),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            rank = TD.rank(G)
            peer = 1 - rank
            if get:
                TD.signal(TD.remote(Ready, peer)[0], epoch0 + signal_offset, channel=CH, release_regions=[Send[:]])
                TD.wait_signal(Ready[0], epoch0 + signal_offset + wait_offset, channel=CH, covers=[TD.remote(Send, peer)[:]])
                event = TD.copy_async(TD.remote(Send, peer)[:], Recv[:], channel=CH)
                TD.wait(event, completion="local")
                TD.signal(TD.remote(Ack, peer)[0], epoch0 + signal_offset, channel=CH)
                T.copy(Recv[:], Output[:])
            else:
                event = TD.copy_async(Send[:], TD.remote(Recv, peer)[:], channel=CH)
                TD.wait(event, completion="local")
                TD.signal(TD.remote(Ready, peer)[0], epoch0 + signal_offset, channel=CH)
                TD.wait_signal(Ready[0], epoch0 + signal_offset + wait_offset, channel=CH, covers=[Recv[:]])
                if partial_overlap:
                    T.copy(Recv[1:5], Output[1:5])
                else:
                    T.copy(Recv[:], Output[:])
                TD.signal(TD.remote(Ack, peer)[0], epoch0 + signal_offset, channel=CH, release_regions=[Recv[:]])
            TD.wait_signal(Ack[0], epoch0 + signal_offset + wait_offset, channel=CH, covers=[])
            TD.fence(channel=CH)

    return main


def setup(program, tmp_path):
    helper = os.environ.get("CANNSIM_ROOT")
    if not helper or not (Path(helper) / "cosim/protocol.py").is_file():
        pytest.skip("set CANNSIM_ROOT to the extended local cannsim-helper")
    compiled = D.compile(program, mode="protocol", helper_root=helper, output_dir=tmp_path)
    session = D.init(world_size=2, profile="ascend950_cannsim_v1")
    for spec in compiled.manifest["objects"]:
        if spec["kind"] == "signal":
            D.alloc_signals(session, key=spec["key"], shape=spec["shape"])
        else:
            D.alloc_symmetric(session, key=spec["key"], shape=spec["shape"], dtype=spec["dtype"], remote_access=spec["remote_access"])
    ch = D.channel(session, key="comm")
    return compiled, session, ch


@pytest.mark.parametrize("get", [False, True])
def test_ir_protocol_put_get_and_two_launch_epochs(tmp_path, get):
    compiled, ctx, ch = setup(exchange(get=get), tmp_path)
    for iteration in range(2):
        ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="exchange", parameters={})
        source = {rank: np.arange(8, dtype=np.float32) + rank * 10 + iteration * 100 for rank in range(2)}
        D.copy_in(ctx.objects["Send"], source, ticket=ticket)
        run = D.launch(compiled, ticket=ticket)
        report = D.complete(run)
        assert report["passed"] and not report["kernel_executed"]
        assert report["epoch_base"] == iteration
        assert not report["protocol"]["pending_events"]
        assert not report["protocol"]["open_leases"]
        for rank in range(2):
            np.testing.assert_array_equal(ctx.objects["Output"].arrays[rank], source[1 - rank])
        saved = json.loads((Path(report["run_dir"]) / "abi.json").read_text())
        assert saved["program_sha256"] == compiled.manifest["program_sha256"]
    D.close(ctx)


def test_partial_alias_is_rejected_before_protocol_execution(tmp_path):
    compiled, ctx, ch = setup(exchange(partial_overlap=True), tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="exchange", parameters={})
    with pytest.raises(RuntimeError, match="failed"):
        D.launch(compiled, ticket=ticket)
    report = ctx._run.result
    assert not report["passed"]
    assert "partially overlapping" in report["errors"][0]
    assert report["protocol"]["events"] == 0


@pytest.mark.parametrize("fault", ["drop_signal", "drop_ack"])
def test_ir_fault_propagates_and_preserves_report(tmp_path, fault):
    compiled, ctx, ch = setup(exchange(), tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="exchange", parameters={})
    with pytest.raises(RuntimeError, match="failed"):
        D.launch(compiled, ticket=ticket, fault=fault)
    assert ctx.state == "FAILED"
    report = ctx._run.result
    assert report["protocol"]["fault_injected"]
    assert not report["passed"]
    assert (Path(report["run_dir"]) / "protocol_trace.json").is_file()


def test_native_transport_is_rejected_before_ascend_scheduling():
    with pytest.raises(ValueError, match="transport is not certified"):
        tilelang.lower(exchange(), target="ascend")


def test_program_identity_includes_communication_direction(tmp_path):
    put, _, _ = setup(exchange(), tmp_path / "put")
    get, _, _ = setup(exchange(get=True), tmp_path / "get")
    assert put.manifest["program_sha256"] != get.manifest["program_sha256"]


def assert_preflight_rejection(compiled, ctx, ticket, diagnostic):
    with pytest.raises(D.RunFailedError) as caught:
        D.launch(compiled, ticket=ticket)
    report = caught.value.run.result
    assert not report["passed"]
    assert diagnostic in "\n".join(report["errors"])
    assert report["protocol"]["events"] == 0
    assert not report["segments"]


@pytest.mark.parametrize(
    ("signal_offset", "wait_offset", "diagnostic"),
    [
        (0, 0, "published epoch 0"),
        (2, 0, "published epoch 2"),
        (1, 1, "wait epoch 2"),
    ],
)
def test_epoch_ticket_is_checked_before_any_request(tmp_path, signal_offset, wait_offset, diagnostic):
    compiled, ctx, ch = setup(exchange(signal_offset=signal_offset, wait_offset=wait_offset), tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="exchange")
    assert_preflight_rejection(compiled, ctx, ticket, diagnostic)


def test_peer_must_remain_in_prepared_channel(tmp_path):
    compiled, ctx, ch = setup(exchange(), tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="exchange")
    ch.peers[0] = ()
    assert_preflight_rejection(compiled, ctx, ticket, "outside prepared channel")


def scoped_signal(scope):
    @T.prim_func
    def main(Ready: TD.SignalBuffer((1,), group=G), epoch0: T.int32):
        with T.Kernel(1):
            peer = 1 - TD.rank(G)
            if scope == "stage":
                with T.Stage(0):
                    TD.signal(TD.remote(Ready, peer)[0], epoch0 + 1, channel=CH)
            else:
                with T.PerCoreTask():
                    TD.signal(TD.remote(Ready, peer)[0], epoch0 + 1, channel=CH)
            TD.quiet(channel=CH)

    return main


@pytest.mark.parametrize(("scope", "attribute"), [("stage", "tl.ascend_stage"), ("per_core", "tl.ascend_per_core_task")])
def test_real_scheduling_scopes_are_not_silently_erased(tmp_path, scope, attribute):
    compiled, ctx, ch = setup(scoped_signal(scope), tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="scoped-signal")
    assert_preflight_rejection(compiled, ctx, ticket, attribute)


@pytest.mark.parametrize(
    ("block_kind", "diagnostic"),
    [
        ("false_predicate", "unconditional root/allocation blocks"),
        ("iteration", "without block iteration bindings"),
        ("match_alias", "match_buffer aliases"),
    ],
)
def test_nontrivial_blocks_are_rejected_without_ignoring_their_effects(tmp_path, block_kind, diagnostic):
    program = exchange()
    predicate, iter_vars, iter_values, matches = True, [], [], []
    if block_kind == "false_predicate":
        predicate = False
    elif block_kind == "iteration":
        axis = tirx.IterVar(tvm.ir.Range.from_min_extent(0, 1), tirx.Var("block_axis", "int32"), tirx.IterVar.DataPar)
        iter_vars, iter_values = [axis], [tirx.IntImm("int32", 0)]
    else:
        source = program.buffer_map[program.params[0]]
        alias = tirx.decl_buffer((8,), "float32", name="alias")
        matches = [tirx.MatchBufferRegion(alias, tirx.BufferRegion(source, [tvm.ir.Range.from_min_extent(0, 8)]))]
    block = tirx.SBlock(iter_vars, [], [], "guarded", program.body, match_buffers=matches)
    program = program.with_body(tirx.SBlockRealize(iter_values, predicate, block))
    compiled, ctx, ch = setup(program, tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="guarded-exchange")
    assert_preflight_rejection(compiled, ctx, ticket, diagnostic)


def local_copy(mode):
    dtype = "float16" if mode == "cast" else "float32"

    @T.prim_func
    def main(
        Source: TD.SymBuffer((8,), "float32", group=G, remote_access="read"),
        Destination: TD.SymBuffer((8,), dtype, group=G, remote_access="write"),
    ):
        with T.Kernel(1):
            if mode == "transpose":
                T.copy(Source, Destination, transpose=True)
            else:
                T.copy(Source, Destination)
            TD.quiet(channel=CH)

    if mode == "count":

        def change_count(node):
            if isinstance(node, tirx.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.ascend_copy":
                target = node.args[1]
                smaller = tirx.Call(target.dtype, target.op, [*target.args[:-1], tirx.IntImm("int32", 4)], target.annotations)
                return tirx.Call(node.dtype, node.op, [node.args[0], smaller], node.annotations)

        main = main.with_body(tirx.stmt_functor.ir_transform(main.body, None, change_count))
    return main


@pytest.mark.parametrize("mode", ["transpose", "cast", "count"])
def test_local_copy_contract_is_not_replaced_with_raw_numpy_copy(tmp_path, mode):
    compiled, ctx, ch = setup(local_copy(mode), tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="local-copy")
    assert_preflight_rejection(compiled, ctx, ticket, "equal dtype/count and default copy semantics")


@pytest.mark.parametrize("alias_kind", ["objects", "ranks"])
def test_model_never_hides_backing_storage_aliases(tmp_path, alias_kind):
    compiled, ctx, ch = setup(exchange(), tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="exchange")
    if alias_kind == "objects":
        ctx.objects["Output"].arrays[0] = ctx.objects["Recv"].arrays[0]
    else:
        ctx.objects["Send"].arrays[1] = ctx.objects["Send"].arrays[0]
    assert_preflight_rejection(compiled, ctx, ticket, "aliased allocations")


@pytest.mark.parametrize("change", ["dtype", "shape", "signal_stride"])
def test_actual_storage_must_match_prepared_buffer_contract(tmp_path, change):
    compiled, ctx, ch = setup(exchange(), tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="exchange")
    if change == "dtype":
        ctx.objects["Send"].arrays[1] = ctx.objects["Send"].arrays[1].view(np.int32)
    elif change == "shape":
        ctx.objects["Send"].arrays[1] = ctx.objects["Send"].arrays[1].reshape(2, 4)
    else:
        # Rank zero still matches metadata; every rank's backing is checked.
        ctx.objects["Ready"].arrays[1] = np.zeros((1,), dtype=np.int32)
    diagnostic = "symmetric strides" if change == "signal_stride" else "storage dtype/shape"
    assert_preflight_rejection(compiled, ctx, ticket, diagnostic)


def test_loop_annotations_are_not_ignored(tmp_path):
    program = exchange()

    def annotate(node):
        if isinstance(node, tirx.For):
            return tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body, node.thread_binding, {"unvalidated_schedule": 1})

    program = program.with_body(tirx.stmt_functor.ir_transform(program.body, None, annotate))
    compiled, ctx, ch = setup(program, tmp_path)
    ticket = D.prepare_launch(ctx, compiled, channel=ch, epochs=1, protocol="exchange")
    assert_preflight_rejection(compiled, ctx, ticket, "annotated loops")


def test_phase_scalar_arithmetic_preserves_signed_division_and_rejects_narrowing():
    from tilelang.ascend.distributed.cosim import _eval

    value = tirx.Var("value", "int32")
    assert _eval(tirx.Div(value, 2), {value: -3}, 0) == -1
    assert _eval(tirx.Mod(value, 2), {value: -3}, 0) == -1
    with pytest.raises(ValueError, match="overflow or narrowing"):
        _eval(value + 1, {value: 2**31 - 1}, 0)
    with pytest.raises(ValueError, match="scalar integer"):
        _eval(value, {value: 0.5}, 0)
