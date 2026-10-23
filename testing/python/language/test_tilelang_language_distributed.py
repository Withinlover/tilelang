"""Source IR and ABI tests for distributed P0; no NPU is required."""

from dataclasses import FrozenInstanceError
import json
import subprocess
import sys

import pytest
import tilelang  # noqa: F401
import tvm
from tvm import tirx

import tilelang.language as T
import tilelang.language.distributed as TD


GROUP = TD.GroupSpec("world", 2)
CHANNEL = TD.ChannelSpec("comm", GROUP)


def make_exchange(mode="valid"):
    @T.prim_func
    def exchange(
        Send: TD.SymBuffer((2, 8), "float32", group=GROUP, remote_access="read"),
        Recv: TD.SymBuffer((2, 8), "float32", group=GROUP, remote_access="write"),
        Ready: TD.SignalBuffer((2,), group=GROUP),
        Ack: TD.SignalBuffer((2,), group=GROUP),
    ):
        r = TD.rank(GROUP)
        p = TD.size(GROUP) - r - 1
        if mode == "both_local":
            event = TD.copy_async(Send, Recv, channel=CHANNEL)
        if mode == "both_remote":
            event = TD.copy_async(TD.remote(Send, p), TD.remote(Recv, p), channel=CHANNEL)
        if mode == "bad_permission":
            event = TD.copy_async(Send, TD.remote(Send, p), channel=CHANNEL)
        if mode == "strided":
            event = TD.copy_async(Send[:, :4], TD.remote(Recv, p)[:, :4], channel=CHANNEL)
        if mode == "empty":
            event = TD.copy_async(Send[0, :0], TD.remote(Recv, p)[0, :0], channel=CHANNEL)
        if mode == "out_of_bounds":
            event = TD.copy_async(Send[0, :9], TD.remote(Recv, p)[0, :9], channel=CHANNEL)
        if mode == "peer_mismatch":
            event = TD.put_signal(Send[0, :], TD.remote(Recv, p)[0, :], signal=TD.remote(Ready, r)[0], value=1, channel=CHANNEL)
        if mode == "signal_as_payload":
            event = TD.copy_async(Ready, TD.remote(Ack, p), channel=CHANNEL)
        if mode == "get_remote_wait":
            event = TD.copy_async(TD.remote(Send, p)[0, :], Recv[0, :], channel=CHANNEL)
            TD.wait(event, completion="remote")
        if mode == "event_arithmetic":
            event = TD.copy_async(Send[0, :], TD.remote(Recv, p)[0, :], channel=CHANNEL)
            T.evaluate(event + 1)
        if mode == "event_conditional":
            event = TD.copy_async(Send[0, :], TD.remote(Recv, p)[0, :], channel=CHANNEL)
            if r == 0:
                TD.wait(event, completion="local")
        if mode == "signal_read":
            T.evaluate(Ready[0])
        if mode == "signal_write":
            Ready[0] = 1
        if mode == "signal_copy":
            T.copy(Ready, Ack)
        if mode == "signal_pointer":
            T.evaluate(T.call_extern("int32", "external", Ready.data))
        if mode == "valid":
            event = TD.put_signal(Send[0, :], TD.remote(Recv, p)[0, :], signal=TD.remote(Ready, p)[0], value=1, channel=CHANNEL)
            TD.wait(event, completion="local")
            TD.wait(event, completion="remote")
            TD.wait_signal(Ready[0], 1, channel=CHANNEL, covers=[Recv[0, :]])
            TD.signal(TD.remote(Ack, p)[0], 1, channel=CHANNEL, release_regions=[Recv[0, :]])
            TD.fence(channel=CHANNEL)
            TD.quiet(channel=CHANNEL)

    return exchange


def calls(func):
    result = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: result.append(node) if isinstance(node, tirx.Call) else None)
    return [call for call in result if call.op.name.startswith("tl.distributed.")]


def test_protocol_ops_preserve_regions_and_effects():
    func = make_exchange()
    operations = calls(func)
    assert [call.op.name.rsplit(".", 1)[-1] for call in operations] == [
        "rank",
        "put_signal",
        "wait",
        "wait",
        "wait_signal",
        "signal",
        "fence",
        "quiet",
    ]
    put = operations[1]
    assert isinstance(put.annotations["src_region"], tirx.BufferRegion)
    assert put.annotations["src_region"].buffer.same_as(func.buffer_map[func.params[0]])
    assert int(put.annotations["dst_region"].region[1].extent) == 8
    assert all(int(call.op.get_attr("TCallEffectKind")) == tirx.CallEffectKind.Opaque.value for call in operations)
    assert len(operations[4].annotations["covers"]) == 1
    assert len(operations[5].annotations["release_regions"]) == 1


def test_descriptor_and_signal_layout_are_normalized_before_compilation():
    func = make_exchange()
    manifest = json.loads(str(func.attrs[TD.MANIFEST_ATTR]))
    assert len(func.params) == 5
    descriptor = func.buffer_map[func.params[-1]]
    assert descriptor.dtype == "uint8"
    assert list(map(int, descriptor.shape)) == [64 + 4 * 2 * 32]
    assert manifest["descriptor"]["param_index"] == 4
    assert manifest["descriptor"]["encoding"] == "cpu-model-v1"
    signal = func.buffer_map[func.params[2]]
    assert list(map(int, signal.strides)) == [16]
    assert manifest["objects"][2]["physical_nbytes"] == 128
    assert func.buffer_map[func.params[0]].scope() == "global"
    assert "0x" not in str(func.attrs[TD.MANIFEST_ATTR])


def test_manifest_is_stable_and_builder_state_is_isolated():
    first, second = make_exchange(), make_exchange()
    assert str(first.attrs[TD.MANIFEST_ATTR]) == str(second.attrs[TD.MANIFEST_ATTR])
    assert int(calls(first)[1].args[0]) == int(calls(second)[1].args[0]) == 0

    @T.prim_func
    def local(A: T.Tensor((8,), "float32")):
        A[0] = 1.0

    assert len(local.params) == 1
    assert TD.MANIFEST_ATTR not in local.attrs


def test_descriptor_preserves_returned_output_indices():
    @T.prim_func
    def output(A: TD.SymBuffer((8,), "float32", group=GROUP, remote_access="read")):
        result = T.empty((8,), "float32")
        T.copy(A, result)
        return result

    assert len(output.params) == 3
    assert list(map(int, output.attrs["tilelang_out_idx"])) == [-2]


def test_native_ops_and_region_metadata_survive_fresh_process_json(tmp_path):
    path = tmp_path / "distributed.json"
    path.write_text(tvm.ir.save_json(make_exchange()))
    subprocess.run(
        [sys.executable, "-c", "import tilelang,tvm,sys; f=tvm.ir.load_json(open(sys.argv[1]).read()); assert len(f.params)==5", str(path)],
        check=True,
    )


@pytest.mark.parametrize(
    "mode,message",
    [
        ("both_local", "exactly one remote"),
        ("both_remote", "exactly one remote"),
        ("bad_permission", "permission"),
        ("strided", "contiguous"),
        ("empty", "nonempty"),
        ("out_of_bounds", "out of bounds"),
        ("peer_mismatch", "same peer"),
        ("signal_as_payload", "signal APIs"),
        ("get_remote_wait", "only local completion"),
        ("event_arithmetic", "unsupported operand"),
        ("event_conditional", "issuing structured"),
        ("signal_read", "only accepts signal APIs"),
        ("signal_write", "only accepts signal APIs"),
        ("signal_copy", "only accepts signal APIs"),
        ("signal_pointer", "pointer escape"),
    ],
)
def test_invalid_source_contracts(mode, message):
    with pytest.raises((TypeError, ValueError), match=message):
        make_exchange(mode)


@pytest.mark.parametrize("dtype", ["int64", "uint32", "uint64", "float32"])
def test_signal_dtype_is_not_silently_changed(dtype):
    with pytest.raises(ValueError, match="only int32"):
        TD.SignalBuffer((2,), dtype, group=GROUP)


def test_specs_are_frozen_and_runtime_free():
    with pytest.raises(FrozenInstanceError):
        GROUP.size = 4
    with pytest.raises(ValueError, match="positive int32"):
        TD.GroupSpec("bad", 0)
    with pytest.raises(ValueError, match="positive static shape"):
        TD.SymBuffer((0,), "float32", group=GROUP, remote_access="read")
