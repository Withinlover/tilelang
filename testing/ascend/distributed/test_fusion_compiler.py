"""Host-only scheduling regressions for the fused SHMEM completion contract."""

import pytest

import tilelang
from tilelang.ascend import language as T, transform
from tilelang.ascend.distributed.fusion_checks import verify_fused_inventory, verify_fused_tasks
from testing.ascend._ir import calls, copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from tvm import tirx


def ptr(buffer, mask, extent=None):
    return tirx.call_intrin(
        "handle", tirx.op.Op.get("tl.access_ptr"), buffer[tuple(0 for _ in buffer.shape)], extent or int(buffer.shape[0]), mask
    )


def task(kind, scratch, *, effect=None, mask=3, nbytes=0):
    args = [kind, 0, 0, 0, ptr(scratch, mask), nbytes, 1, 1]
    if effect is not None:
        args.append(ptr(effect, 2))
    return tirx.Evaluate(
        tirx.Call("int32", tirx.op.Op.get("tl.ascend_distributed_task"), args, annotations={"td.task_id": tirx.IntImm("int32", kind)})
    )


def test_acquire_and_ack_wait_for_actual_dma_completion():
    recv = tirx.decl_buffer((64,), "float16", name="recv")
    l1 = tirx.decl_buffer((64,), "float16", name="l1", scope="shared.l1")
    scratch = tirx.decl_buffer((4096,), "uint8", name="scratch", scope="shared")
    before = kernel(
        seq(unit(task(4, scratch, effect=recv), core=1), unit(copy(recv, l1), core=2), unit(task(3, scratch, effect=recv), core=1)),
        buffers=[l1, scratch],
        params=[recv],
    )
    after = transform.InsertSync()(before)
    sets = calls(after, "tl.ascend_cross_core_set_flag")
    waits = calls(after, "tl.ascend_cross_core_wait_flag")
    # READY becomes a Scalar->MTE2 cross-core RAW edge. ACK carries a write
    # effect so its WAR edge waits for AIC MTE2 completion, not merely issue.
    assert {call.args[1].value for call in sets} >= {"PIPE_S", "PIPE_MTE2"}
    assert {call.args[1].value for call in waits} >= {"PIPE_S", "PIPE_MTE2"}
    assert len(sets) == len(waits) == 2


@pytest.mark.parametrize("flags", [1, 8], ids=["compact-reserved-slot", "spill-eighth-slot"])
def test_shmem_event_slot_is_excluded_from_flag_allocator(flags):
    scratch = tirx.decl_buffer((4096,), "uint8", name="scratch", scope="shared")
    statements = []
    for event in range(8 - flags, 8):
        for operation in ("set", "wait"):
            statements.append(
                tirx.Evaluate(tirx.call_intrin("int32", tirx.op.Op.get("tl.ascend_" + operation + "_flag"), "MTE2_MTE3", event))
            )
    statements.append(task(5, scratch))
    rewritten = transform.RewriteFlagToBuf()(kernel(seq(*statements), buffers=[scratch]))
    for operation in ("set", "wait"):
        assert all(int(call.args[1]) < 7 for call in calls(rewritten, "tl.ascend_" + operation + "_flag"))
    assert bool(calls(rewritten, "tl.ascend_get_buf")) == (flags == 8)


@pytest.mark.parametrize("scope, size, mask", [("shared", 4096, 1), ("shared", 8192, 3), ("shared.l1", 4096, 3)])
def test_malformed_scratch_contract_is_rejected(scope, size, mask):
    scratch = tirx.decl_buffer((size,), "uint8", name="scratch", scope=scope)
    with pytest.raises(ValueError, match="scratch"):
        verify_fused_tasks(kernel(task(5, scratch, mask=mask), buffers=[scratch]))


def test_fused_path_requires_scheduler_and_preserves_task_inventory():
    scratch = tirx.decl_buffer((4096,), "uint8", name="scratch", scope="shared")
    before = kernel(task(5, scratch), buffers=[scratch])
    expected = verify_fused_tasks(before)
    with pytest.raises(ValueError, match="requires AutoSchedule"):
        verify_fused_tasks(before, auto_schedule=False)
    with pytest.raises(ValueError, match="deleted or duplicated"):
        verify_fused_inventory(kernel(tirx.Evaluate(0)), expected)


@pytest.mark.parametrize("blocks, sids", [(2, 1), (1, 2)])
def test_multiple_issuers_are_rejected(blocks, sids):
    @T.prim_func
    def source():
        with T.MixedKernel(blocks, sids=sids):
            scratch = T.alloc_shared((4096,), "uint8")
            T.evaluate(
                T.call_intrin(
                    "int32",
                    tirx.op.Op.get("tl.ascend_distributed_task"),
                    5,
                    0,
                    0,
                    0,
                    T.access_ptr(scratch[0], "rw", extent=4096),
                    0,
                    1,
                    1,
                    annotations={"td.task_id": 0},
                )
            )

    with pytest.raises(ValueError, match="one block and exactly one active AIV"):
        tilelang.lower(source, target={"kind": "ascend", "arch": "dav-3510"})


@pytest.mark.parametrize("change", ["epoch", "effect", "order"])
def test_terminal_verifier_rejects_protocol_tampering(change):
    scratch = tirx.decl_buffer((4096,), "uint8", name="scratch", scope="shared")
    recv = tirx.decl_buffer((64,), "float16", name="recv")
    first = task(4, scratch, effect=recv)
    last = task(5, scratch)
    before = kernel(seq(first, last), buffers=[scratch], params=[recv])
    expected = verify_fused_tasks(before)
    args = list(first.value.args)
    if change == "epoch":
        args[7] = tirx.const(2, "int32")
    elif change == "effect":
        args.pop()
    changed = tirx.Evaluate(tirx.Call("int32", first.value.op, args, annotations=first.value.annotations))
    statements = [last, changed] if change == "order" else [changed, last]
    with pytest.raises(ValueError, match="epoch|effect operands|reordered"):
        verify_fused_inventory(kernel(seq(*statements), buffers=[scratch], params=[recv]), expected)


def test_scratch_cannot_be_borrowed_for_scalar_storage():
    scratch = tirx.decl_buffer((4096,), "uint8", name="scratch", scope="shared")
    before = kernel(seq(tirx.BufferStore(scratch, tirx.const(0, "uint8"), [0]), task(5, scratch)), buffers=[scratch])
    with pytest.raises(ValueError, match="cannot be borrowed"):
        verify_fused_tasks(before)


def test_allgather_ack_releases_dma_without_waiting_for_gemm():
    from examples.distributed.example_allgather_gemm import allgather_gemm
    from tilelang.ascend.distributed.fused import FusedCompiledProgram
    from tilelang.distributed.abi import P0B_PROFILE
    from testing.ascend._ir import nodes, schedule

    compiled = FusedCompiledProgram(allgather_gemm(2, 1), profile=P0B_PROFILE)
    compiled.plan()
    scheduled = schedule(compiled._funcs()[0], through="LowerScheduledTIR", infer_layouts=True)
    blocks = nodes(scheduled, tirx.SBlock)
    vector = next(block for block in blocks if block.name_hint == "VECTOR")
    cube = next(block for block in blocks if block.name_hint == "CUBE")
    vector_calls = nodes(vector.body, tirx.Call)
    cube_calls = nodes(cube.body, tirx.Call)
    ack_positions = [i for i, call in enumerate(vector_calls) if call.op.name == "tl.ascend_distributed_task" and int(call.args[0]) == 3]
    assert len(ack_positions) == 2
    for position in ack_positions:
        # Each ACK's last local cross-core wait is fed by the MTE2 copy of
        # Recv. Its producer signal is issued before the corresponding GEMM.
        wait = next(call for call in reversed(vector_calls[:position]) if call.op.name == "tl.ascend_cross_core_wait_flag")
        assert wait.args[1].value == "PIPE_S"
        producer_pos = next(
            i
            for i, call in enumerate(cube_calls)
            if call.op.name == "tl.ascend_cross_core_set_flag" and int(call.args[2]) == int(wait.args[2])
        )
        assert cube_calls[producer_pos].args[1].value == "PIPE_MTE2"
        preceding_copy = next(call for call in reversed(cube_calls[:producer_pos]) if call.op.name == "tl.tileop.ascend_copy")
        assert preceding_copy.args[0].args[0].buffer.name == "Recv"
        assert any(call.op.name == "tl.tileop.gemm" for call in cube_calls[producer_pos + 1 :])
    # No GEMM/FIX completion is in the AIV dependency stream; the next put is
    # reachable after ACK/peer-ACK while Cube executes its resident tile.
    assert {call.args[1].value for call in calls(cube.body, "tl.ascend_cross_core_set_flag")} == {"PIPE_MTE2"}
    between_acks = vector_calls[ack_positions[0] + 1 : ack_positions[1]]
    next_put = next(i for i, call in enumerate(between_acks) if call.op.name == "tl.ascend_distributed_task" and int(call.args[0]) == 2)
    next_cube_wait = next(i for i, call in enumerate(between_acks) if call.op.name == "tl.ascend_cross_core_wait_flag")
    assert next_put < next_cube_wait


def test_final_storage_check_detects_local_access_into_comm_scratch():
    from examples.distributed.example_fused_exchange import vector_exchange
    from tilelang import tvm
    from tilelang.ascend.distributed.fused import FusedCompiledProgram
    from tilelang.ascend.distributed.fusion_checks import verify_fused_storage

    compiled = FusedCompiledProgram(vector_exchange(1, 1))
    compiled.plan()
    lowered = tilelang.lower(compiled._funcs()[0], target={"kind": "ascend", "arch": "dav-3510"}).device_mod
    global_var, function = next(iter(lowered.functions.items()))
    scratch = calls(function, "tl.ascend_distributed_task")[0].args[4]
    changed = []

    def corrupt(node):
        if (
            not changed
            and isinstance(node, tirx.Call)
            and node.op.name == "tirx.tvm_access_ptr"
            and node.args[1].same_as(scratch.args[1])
            and node.args[0].dtype == "float32"
        ):
            args = list(node.args)
            args[2] = scratch.args[2] // 4
            changed.append(True)
            return tirx.Call(node.dtype, node.op, args, annotations=node.annotations)

    body = tirx.stmt_functor.ir_transform(function.body, corrupt, None)
    assert changed
    with pytest.raises(ValueError, match="overlaps the reserved physical"):
        verify_fused_storage(tvm.IRModule({global_var: function.with_body(body)}))


@pytest.mark.parametrize("reuse_storage", [True, False], ids=["completion-return-edge", "issue-order-only"])
def test_straight_line_flags_reuse_requires_completion_order(reuse_storage):
    count = 10
    source = tirx.decl_buffer((count * 64,), "float32", name="source")
    destination = tirx.decl_buffer((count * 64,), "float32", name="destination")
    buffers = [tirx.decl_buffer((64,), "float32", name=f"ub{i}", scope="shared.dyn") for i in range(1 if reuse_storage else count)]
    operations = []
    for i in range(count):
        ub = buffers[0 if reuse_storage else i]
        operations.extend(
            [
                unit(copy(source, ub, src_indices=[i * 64], src_shape=[64])),
                unit(copy(ub, destination, dst_indices=[i * 64], dst_shape=[64])),
            ]
        )
    scheduled = transform.InsertSync()(kernel(seq(*operations), buffers=buffers, params=[source, destination]))
    set_ids = {int(call.args[1]) for call in calls(scheduled, "tl.ascend_set_flag") if call.args[0].value == "MTE2_MTE3"}
    wait_ids = {int(call.args[1]) for call in calls(scheduled, "tl.ascend_wait_flag") if call.args[0].value == "MTE2_MTE3"}
    assert set_ids == wait_ids
    if reuse_storage:
        # The MTE3->MTE2 return dependency proves that an earlier consumer has
        # finished before the next producer can publish into a reused event.
        assert len(set_ids) <= 8
        assert not calls(transform.RewriteFlagToBuf()(scheduled), "tl.ascend_get_buf")
    else:
        # Lexical SET/WAIT order on independent DMA pipes is insufficient.
        # All ten events must remain distinct and two must spill.
        assert len(set_ids) == count
        assert calls(transform.RewriteFlagToBuf()(scheduled), "tl.ascend_get_buf")


def test_p0b_seven_round_hot_slot_fits_physical_event_capacity():
    from examples.distributed.example_allgather_gemm import allgather_gemm
    from tilelang.ascend.distributed.fused import FusedCompiledProgram
    from tilelang.distributed.abi import P0B_PROFILE

    compiled = FusedCompiledProgram(allgather_gemm(7, 1), profile=P0B_PROFILE)
    compiled.plan()
    artifact = tilelang.lower(compiled._funcs()[0], target={"kind": "ascend", "arch": "dav-3510"})
    function = next(iter(artifact.device_mod.functions.values()))
    for operation in ("set", "wait"):
        assert all(int(call.args[1]) < 7 for call in calls(function, "tl.ascend_" + operation + "_flag"))
    assert artifact.kernel_source.count("__global__") == 1
