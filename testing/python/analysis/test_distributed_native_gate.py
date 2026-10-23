"""Native execution never silently accepts unimplemented communication IR."""

import importlib

import pytest
import tilelang
import tilelang.language as T
import tilelang.language.distributed as TD
from tilelang.analysis.distributed_checker import reject_native_distributed
from tilelang.cache import cached
from tilelang.jit.kernel import JITKernel
from tvm import IRModule, tirx


G = TD.GroupSpec("world", 2)
C = TD.ChannelSpec("comm", G)
MESSAGE = "native transport is not certified.*tilelang.distributed.compile"


@pytest.fixture
def distributed():
    @T.prim_func
    def exchange(
        Source: TD.SymBuffer((8,), "float32", group=G, remote_access="read"),
        Destination: TD.SymBuffer((8,), "float32", group=G, remote_access="write"),
    ):
        rank = TD.rank(G)
        event = TD.copy_async(Source, TD.remote(Destination, 1 - rank), channel=C)
        TD.wait(event, completion="local")
        TD.quiet(channel=C)

    return exchange


@pytest.mark.parametrize("entry", [tilelang.compile, cached, JITKernel, tilelang.lower])
@pytest.mark.parametrize("target", ["cuda", "c", "ascend", "auto"])
def test_native_public_entries_gate_before_target_resolution(distributed, entry, target):
    with pytest.raises(ValueError, match=MESSAGE):
        entry(distributed, target=target)


@pytest.mark.parametrize("representation", ["manifest_only", "ops_only", "module"])
def test_gate_detects_ir_module_and_either_representation(distributed, representation):
    if representation == "manifest_only":
        program = distributed.with_body(tirx.Evaluate(0))
    elif representation == "ops_only":
        program = distributed.without_attr(TD.MANIFEST_ATTR)
    else:
        # An earlier ordinary function must not hide a later distributed one.
        local = tirx.PrimFunc([], tirx.Evaluate(0))
        program = IRModule({"local": local, "exchange": distributed})
    with pytest.raises(ValueError, match=MESSAGE):
        tilelang.lower(program, target="ascend")


def test_compile_ir_module_is_gated_before_primfunc_type_assertion(distributed):
    with pytest.raises(ValueError, match=MESSAGE):
        tilelang.compile(IRModule({"exchange": distributed}), target="cuda")


@pytest.mark.parametrize("entry", ["lower_with_context", "lower_to_host_device_ir"])
def test_context_lowering_gates_before_context_or_parameter_access(distributed, entry):
    module = importlib.import_module("tilelang.engine.lower")
    with pytest.raises(ValueError, match=MESSAGE):
        getattr(module, entry)(IRModule({"exchange": distributed}), context=None)


def test_semantic_optout_and_database_loading_cannot_disable_capability_gate(distributed, monkeypatch):
    disabled = {tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK: True}
    with pytest.raises(ValueError, match=MESSAGE):
        tilelang.compile(distributed, target="ascend", pass_configs=disabled)
    with pytest.raises(ValueError, match=MESSAGE):
        JITKernel(distributed, target="cuda", pass_configs=disabled, from_database=True)
    # Some builds do not register the Python semantic option in native TVM's
    # PassContext registry. Exercise the disabled semantic path directly too.
    semantic = importlib.import_module("tilelang.engine.semantic_check")
    monkeypatch.setattr(semantic, "should_enable_prelower_semantic_check", lambda: False)
    with pytest.raises(ValueError, match=MESSAGE):
        tilelang.lower(distributed, target="ascend")


def test_gate_precedes_cache_dispatch(distributed, monkeypatch):
    cache_module = importlib.import_module("tilelang.cache")

    def unavailable_dispatch(*args, **kwargs):
        raise AssertionError("cache/backend resolution must not run for uncertified native communication")

    monkeypatch.setattr(cache_module, "_resolve_cache_dispatch", unavailable_dispatch)
    with pytest.raises(ValueError, match=MESSAGE):
        tilelang.compile(distributed, target="ascend")
    with pytest.raises(ValueError, match=MESSAGE):
        cached(distributed, target="ascend")


def test_ordinary_local_ir_is_accepted_by_gate():
    local = tirx.PrimFunc([], tirx.Evaluate(0))
    reject_native_distributed(local)
    reject_native_distributed(IRModule({"local": local}))


def test_direct_ascend_pipeline_reuses_the_same_gate(distributed):
    from tilelang.ascend.pipeline import AscendPassPipelineBody

    with pytest.raises(ValueError, match=MESSAGE):
        AscendPassPipelineBody(IRModule({"exchange": distributed.without_attr(TD.MANIFEST_ATTR)}), target=None)
