"""Region ownership and IR-bypass regression tests for P0 communication."""

import json

import pytest
import tilelang  # noqa: F401
import tvm
from tvm import tirx

import tilelang.language as T
import tilelang.language.distributed as TD
from tilelang.analysis.distributed_checker import check_distributed, op_name
from tilelang.ascend.distributed.capabilities import require_cosim_profile


G = TD.GroupSpec("world", 2)
C = TD.ChannelSpec("comm", G)


def protocol(kind="put", action="none", completion="local"):
    @T.prim_func
    def program(
        Source: TD.SymBuffer((16,), "float32", group=G, remote_access="read"),
        Destination: TD.SymBuffer((16,), "float32", group=G, remote_access="write"),
        Ready: TD.SignalBuffer((2,), group=G),
        Out: T.Tensor((16,), "float32"),
    ):
        rank = TD.rank(G)
        if kind == "put":
            event = TD.copy_async(Source[4:8], TD.remote(Destination, 1 - rank)[4:8], channel=C)
        if kind == "get":
            event = TD.copy_async(TD.remote(Source, 1 - rank)[4:8], Destination[4:8], channel=C)
        if action == "wait":
            TD.wait(event, completion=completion)
        if action == "quiet":
            TD.quiet(channel=C)
        if action == "release":
            TD.signal(TD.remote(Ready, 1 - rank)[0], 1, channel=C)
        if action == "fence":
            TD.fence(channel=C)
        if action == "branch_issue":  # noqa: SIM102 - separate compile-time and device branches
            if rank == 0:
                TD.copy_async(Source[8:12], TD.remote(Destination, 1 - rank)[8:12], channel=C)
        if action == "loop_issue":
            for i in T.serial(2):
                TD.copy_async(Source[i * 4 : i * 4 + 4], TD.remote(Destination, 1 - rank)[i * 4 : i * 4 + 4], channel=C)
        if action == "disjoint":
            Source[0] = 1.0
            Out[0] = Destination[0]
        if action == "unknown":
            Source[rank * 8] = 1.0
        if action == "condition":  # noqa: SIM102 - condition must remain a device memory access
            if Destination[5] > 0:
                Out[0] = 1.0
        if action in ("none", "wait", "quiet", "release", "fence"):
            if kind == "put":
                Source[5] = 1.0
            if kind == "get":
                Out[0] = Destination[5]
        TD.quiet(channel=C)

    return program


def rewrite_call(func, target, update):
    def post(node):
        if op_name(node) == f"tl.distributed.{target}":
            args, annotations = list(node.args), dict(node.annotations)
            update(args, annotations)
            return tirx.Call(node.dtype, node.op, args, annotations=annotations)

    body = tirx.stmt_functor.ir_transform(func.body, None, post, ["tirx.Call"])
    return func.with_body(body)


@pytest.mark.parametrize("kind", ["put", "get"])
@pytest.mark.parametrize("action", ["wait", "quiet", "release", "disjoint", "unknown"])
def test_completed_disjoint_and_unproven_accesses_remain_valid(kind, action):
    check_distributed(protocol(kind, action))


@pytest.mark.parametrize("kind", ["put", "get"])
@pytest.mark.parametrize("action", ["none", "fence"])
def test_nonzero_offset_access_requires_completion(kind, action):
    with pytest.raises(ValueError, match="still owns"):
        check_distributed(protocol(kind, action))


def test_condition_reads_are_memory_accesses():
    with pytest.raises(ValueError, match="still owns"):
        check_distributed(protocol("get", "condition"))


@pytest.mark.parametrize("action", ["branch_issue", "loop_issue"])
def test_pending_request_cannot_escape_structured_scope(action):
    with pytest.raises(ValueError, match="escapes a structured"):
        check_distributed(protocol(action=action))


def test_ir_bypass_cannot_write_read_only_symmetric_object():
    func = protocol(action="wait")

    def update(args, annotations):
        annotations["dst_region"] = annotations["src_region"]

    with pytest.raises(ValueError, match="permission"):
        check_distributed(rewrite_call(func, "copy_async", update))


def test_ir_bypass_cannot_change_peer_ownership():
    func = protocol(action="wait")

    def update(args, annotations):
        args[2] = tirx.IntImm("int32", -1)

    with pytest.raises(ValueError, match="exactly one remote"):
        check_distributed(rewrite_call(func, "copy_async", update))


@pytest.mark.parametrize("offset", [-1, 16])
def test_ir_bypass_cannot_escape_payload_bounds(offset):
    func = protocol(action="wait")

    def update(args, annotations):
        region = annotations["dst_region"]
        annotations["dst_region"] = tirx.BufferRegion(region.buffer, [tvm.ir.Range.from_min_extent(offset, 4)])

    with pytest.raises(ValueError, match="out of bounds"):
        check_distributed(rewrite_call(func, "copy_async", update))


def test_ir_bypass_cannot_escape_signal_bounds():
    func = protocol(action="release")

    def update(args, annotations):
        region = annotations["signal_region"]
        annotations["signal_region"] = tirx.BufferRegion(region.buffer, [tvm.ir.Range.from_min_extent(2, 1)])

    with pytest.raises(ValueError, match="out of bounds"):
        check_distributed(rewrite_call(func, "signal", update))


@pytest.mark.parametrize("value", [tirx.IntImm("int32", 0), tirx.IntImm("int64", 1)])
def test_ir_bypass_cannot_change_signal_value_contract(value):
    func = protocol(action="release")

    def update(args, annotations):
        args[1] = value

    with pytest.raises(ValueError, match="signal value"):
        check_distributed(rewrite_call(func, "signal", update))


def test_missing_manifest_does_not_disable_verification():
    with pytest.raises(ValueError, match="canonical ABI manifest"):
        check_distributed(protocol(action="wait").without_attr(TD.MANIFEST_ATTR))


def test_signal_identity_comes_from_manifest_even_without_auxiliary_map():
    func = protocol(action="wait").without_attr(TD.BUFFER_SPECS_ATTR)
    signal = func.buffer_map[func.params[2]]
    body = tirx.SeqStmt([func.body, tirx.BufferStore(signal, tirx.IntImm("int32", 1), [0])])
    with pytest.raises(ValueError, match="ordinary data access"):
        check_distributed(func.with_body(body))


def test_signal_storage_contract_is_checked_against_actual_ir():
    func = protocol(action="wait")
    spec = json.loads(str(func.attrs[TD.MANIFEST_ATTR]))
    spec["objects"][2]["strides"] = [1]
    with pytest.raises(ValueError, match="physical strides"):
        check_distributed(func.with_attr(TD.MANIFEST_ATTR, json.dumps(spec)))


def test_hardware_profile_requires_certification():
    require_cosim_profile("ascend950_cannsim_v1")
    with pytest.raises(ValueError, match="lacks certified peer mapping"):
        require_cosim_profile("ascend950_mapped_mte_v1")
