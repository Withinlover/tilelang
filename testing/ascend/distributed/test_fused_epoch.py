"""Cache eligibility proof: uncertainty always falls back to real planning."""

import pytest

from tvm import ir, tirx

from examples.distributed.example_allgather_gemm import allgather_gemm
from examples.distributed.example_fused_exchange import vector_exchange
from examples.distributed.run_native_distributed import exchange
from tilelang.ascend.distributed.fused_epoch import prove_epoch_translation


def signal(value, *, annotation=None):
    return tirx.Evaluate(tirx.Call("int32", ir.Op.get("tl.distributed.wait_signal"), [value], annotations=annotation or {}))


def program(epoch, *statements, params=()):
    body = statements[0] if len(statements) == 1 else tirx.SeqStmt(list(statements))
    return tirx.PrimFunc([epoch, *params], body)


@pytest.mark.parametrize("factory", [exchange, allgather_gemm, vector_exchange])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("rounds,slots", [(1, 2), (7, 1)])
def test_supported_examples_prove_all_epochs_in_exact_safe_interval(factory, rank, rounds, slots):
    proof = prove_epoch_translation(factory(rounds, slots), rank=rank)
    assert proof is not None
    assert proof.min_base == 0
    assert proof.max_base == 2**31 - 1 - rounds


@pytest.mark.parametrize("fault", ["control", "nonlinear", "address", "computation", "annotation", "narrow"])
def test_epoch_uses_that_base_zero_and_one_sampling_cannot_prove(fault):
    epoch = tirx.Var("epoch0", "int32")
    alias = tirx.Var("e", "int32")
    good = signal(alias)
    if fault == "control":
        body = tirx.IfThenElse(epoch >= 2, signal(alias + 1), good)
    elif fault == "nonlinear":
        body = signal(alias + epoch * (epoch - 1))
    elif fault == "address":
        buffer = tirx.decl_buffer((64,), "float32")
        body = tirx.Evaluate(buffer[alias])
    elif fault == "computation":
        body = tirx.Evaluate(alias + 42)
    elif fault == "annotation":
        buffer = tirx.decl_buffer((64,), "float32")
        region = tirx.BufferRegion(buffer, [ir.Range.from_min_extent(alias, 1)])
        body = signal(alias, annotation={"covers": [region]})
    else:
        body = signal(tirx.Cast("int32", tirx.Cast("int16", alias)))
    assert prove_epoch_translation(program(epoch, tirx.Bind(alias, epoch + 1), body)) is None


@pytest.mark.parametrize("explicit_cast", [False, True])
def test_wide_alias_never_loses_epoch_dependency(explicit_cast):
    epoch = tirx.Var("epoch0", "int32")
    wide = tirx.Var("wide", "int64")
    value = tirx.Cast("int64", epoch) if explicit_cast else epoch
    if not explicit_cast:
        # TIR itself disallows an implicit change of a Bind value's dtype.
        from tvm.error import InternalError

        with pytest.raises(InternalError, match="value.dtype"):
            tirx.Bind(wide, value)
        return
    source = program(epoch, tirx.Bind(wide, value), tirx.Evaluate(wide), signal(epoch + 1))
    assert prove_epoch_translation(source) is None


def test_scalar_specialization_changes_safe_interval_without_epoch_control():
    epoch = tirx.Var("epoch0", "int32")
    offset = tirx.Var("offset", "int32")
    source = program(epoch, tirx.IfThenElse(offset > 2, signal(epoch + offset), signal(epoch + 1)), params=[offset])
    assert prove_epoch_translation(source, scalars={"offset": 1}).max_base == 2**31 - 2
    assert prove_epoch_translation(source, scalars={"offset": 5}).max_base == 2**31 - 6
    assert prove_epoch_translation(source) is None


def test_intermediate_overflow_bounds_survive_final_cancellation():
    epoch = tirx.Var("epoch0", "int32")
    source = program(epoch, signal((epoch + 10) - 9))
    proof = prove_epoch_translation(source)
    assert proof is not None
    assert proof.max_base == 2**31 - 11  # Final epoch+1 alone is insufficient.


def test_intermediate_underflow_also_bounds_epoch_base():
    epoch = tirx.Var("epoch0", "int32")
    source = program(epoch, signal(((epoch - (2**31 - 1)) - 2) + (2**31 - 1)))
    proof = prove_epoch_translation(source)
    assert proof is not None
    assert proof.min_base == 1


@pytest.mark.parametrize("location", ["loop", "predicate", "allocation", "call_annotation"])
def test_hidden_epoch_metadata_and_control_fall_back(location):
    epoch = tirx.Var("epoch0", "int32")
    if location == "loop":
        body = tirx.For(tirx.Var("i", "int32"), 0, epoch + 1, tirx.ForKind.SERIAL, signal(epoch + 1))
    elif location == "predicate":
        body = tirx.SBlockRealize([], epoch > 0, tirx.SBlock([], [], [], "root", signal(epoch + 1)))
    elif location == "allocation":
        buffer = tirx.decl_buffer((epoch + 64,), "float32", scope="shared")
        body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "root", signal(epoch + 1), alloc_buffers=[buffer]))
    else:
        body = signal(epoch + 1, annotation={"hidden": [epoch]})
    assert prove_epoch_translation(program(epoch, body)) is None


def test_mutable_bind_recurrence_cannot_be_treated_as_a_fresh_alias():
    epoch = tirx.Var("epoch0", "int32")
    total = tirx.Var("total", "int32")
    loop = tirx.For(
        tirx.Var("i", "int32"),
        0,
        3,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt([tirx.Bind(total, total + epoch), signal(total)]),
    )
    assert prove_epoch_translation(program(epoch, tirx.Bind(total, tirx.const(0, "int32")), loop)) is None


def test_reused_loop_variable_falls_back_instead_of_rebinding_analyzer():
    epoch = tirx.Var("epoch0", "int32")
    index = tirx.Var("i", "int32")
    loop = tirx.For(index, 0, 2, tirx.ForKind.SERIAL, signal(epoch + index + 1))
    assert prove_epoch_translation(program(epoch, loop, loop)) is None
