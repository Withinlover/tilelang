"""P0 distributed source DSL with explicit regions and completion semantics.

The initial executable profile is local CANN co-simulation. These operations
describe a protocol; they do not claim an available hardware transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Literal

import tvm
from tvm import tirx
from tvm.ir import Range
from tvm.script.ir_builder import tirx as ir_builder

MANIFEST_ATTR = "tl.distributed.manifest"
BUFFER_SPECS_ATTR = "tl.distributed.buffer_specs"
PROFILE = "ascend950_cannsim_v1"
SIGNAL_CELL_BYTES = 64


def _name(value: str, kind: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ValueError(f"TD.{kind}: name must be a nonempty identifier")


@dataclass(frozen=True)
class GroupSpec:
    """Serializable group name and compile-time rank count."""

    name: str
    size: int

    def __post_init__(self):
        _name(self.name, "GroupSpec")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or not 1 <= self.size <= 2**31 - 1:
            raise ValueError("TD.GroupSpec: size must be a positive int32 compile-time integer")


@dataclass(frozen=True)
class ChannelSpec:
    """Serializable channel binding within one group."""

    name: str
    group: GroupSpec

    def __post_init__(self):
        _name(self.name, "ChannelSpec")
        if not isinstance(self.group, GroupSpec):
            raise TypeError("TD.ChannelSpec: group must be a GroupSpec")


@dataclass(frozen=True)
class _BufferSpec:
    group: GroupSpec
    kind: str
    remote_access: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    physical_nbytes: int


def _shape(shape):
    result = tuple(shape)
    if not result or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in result):
        raise ValueError("TD symmetric buffers require a nonempty, positive static shape")
    return result


def _strides(shape, inner=1):
    result = []
    for extent in reversed(shape):
        result.append(inner)
        inner *= extent
    return tuple(reversed(result))


def SymBuffer(shape, dtype="float32", *, group: GroupSpec, remote_access: Literal["read", "write", "read_write"]):
    """Declare an ordinary local GM Buffer with symmetric-object metadata."""
    if not isinstance(group, GroupSpec):
        raise TypeError("TD.SymBuffer: group must be a GroupSpec")
    if remote_access not in ("read", "write", "read_write"):
        raise ValueError("TD.SymBuffer: remote_access must be read, write, or read_write")
    shape = _shape(shape)
    dt = tvm.DataType(dtype)
    if dt.lanes != 1 or dt.bits % 8 or str(dt) == "handle":
        raise ValueError("TD.SymBuffer: P0 requires byte-addressable scalar numeric dtype")
    buffer = tirx.decl_buffer(shape, dtype, name="symmetric", scope="global", data_alignment=64)
    buffer._distributed_spec = _BufferSpec(group, "symmetric", remote_access, shape, _strides(shape), math.prod(shape) * dt.bits // 8)
    return buffer


def SignalBuffer(shape, dtype="int32", *, group: GroupSpec):
    """Declare int32 signals, each occupying an isolated 64-byte cell."""
    if str(dtype) != "int32":
        raise ValueError("TD.SignalBuffer: ascend950_cannsim_v1 supports only int32 signals")
    if not isinstance(group, GroupSpec):
        raise TypeError("TD.SignalBuffer: group must be a GroupSpec")
    shape = _shape(shape)
    strides = _strides(shape, SIGNAL_CELL_BYTES // 4)
    buffer = tirx.decl_buffer(shape, dtype, name="signals", strides=strides, scope="global", data_alignment=64)
    buffer._distributed_spec = _BufferSpec(group, "signal", "write", shape, strides, math.prod(shape) * SIGNAL_CELL_BYTES)
    return buffer


@dataclass
class _State:
    groups: dict[str, GroupSpec] = field(default_factory=dict)
    channels: dict[str, ChannelSpec] = field(default_factory=dict)
    buffers: dict = field(default_factory=dict)
    event_count: int = 0


def _builder():
    from tilelang.language.eager.builder import Builder

    builder = Builder.current()
    if builder is None:
        raise RuntimeError("TD operations must be constructed inside @T.prim_func")
    if not hasattr(builder, "distributed_state"):
        builder.distributed_state = _State()
    return builder


def _group(group):
    if not isinstance(group, GroupSpec):
        raise TypeError("TD group must be a GroupSpec")
    state = _builder().distributed_state
    previous = state.groups.setdefault(group.name, group)
    if previous != group:
        raise ValueError(f"TD group {group.name!r} has inconsistent sizes")
    return state


def _channel(channel):
    if not isinstance(channel, ChannelSpec):
        raise TypeError("TD channel must be a ChannelSpec")
    state = _group(channel.group)
    previous = state.channels.setdefault(channel.name, channel)
    if previous != channel:
        raise ValueError(f"TD channel {channel.name!r} has inconsistent groups")
    return state


def bind_buffer_annotation(builder, original, argument):
    """Transfer annotation metadata to the builder-owned parameter identity."""
    spec = getattr(original, "_distributed_spec", None)
    if spec is not None:
        state = _group(spec.group)
        state.buffers[argument.data] = (argument, spec)
        argument._distributed_spec = spec


def _spec(buffer):
    state = _builder().distributed_state
    pair = state.buffers.get(buffer.data)
    if pair is None:
        raise TypeError(f"TD communication endpoint {buffer.name!r} must be a SymBuffer or SignalBuffer parameter")
    return pair[1]


def _region(value):
    if isinstance(value, tirx.Buffer):
        return tirx.BufferRegion(value, [Range.from_min_extent(0, x) for x in value.shape])
    if isinstance(value, tirx.BufferLoad):
        return tirx.BufferRegion(value.buffer, [Range.from_min_extent(x, 1) for x in value.indices])
    if isinstance(value, tirx.BufferRegion):
        return value
    raise TypeError("TD endpoints must be Buffer, BufferRegion, or RemoteRegion; ordinary pointers are not supported")


def _int32(value, what):
    if isinstance(value, bool):
        raise TypeError(f"TD {what} must be int32")
    if isinstance(value, int):
        if not -(2**31) <= value < 2**31:
            raise ValueError(f"TD {what} exceeds int32 range")
        return tirx.IntImm("int32", value)
    if not isinstance(value, tirx.PrimExpr) or value.dtype != "int32":
        raise TypeError(f"TD {what} must be int32 without implicit narrowing")
    return value


@dataclass(frozen=True, eq=False)
class RemoteRegion:
    """Non-dereferenceable reference to another rank's symmetric region."""

    region: tirx.BufferRegion
    peer: tirx.PrimExpr
    spec: _BufferSpec

    def __getitem__(self, key):
        raise TypeError("TD RemoteRegion cannot be dereferenced; use TD.copy_async or TD.put_signal")

    def __bool__(self):
        raise TypeError("TD RemoteRegion is an opaque communication endpoint")


@dataclass(frozen=True, eq=False)
class RemoteView:
    buffer: tirx.Buffer
    peer: tirx.PrimExpr
    spec: _BufferSpec

    def __getitem__(self, indices):
        return RemoteRegion(_region(self.buffer[indices]), self.peer, self.spec)

    def __setitem__(self, indices, value):
        raise TypeError("TD remote stores require TD.copy_async or TD.put_signal")

    def __bool__(self):
        raise TypeError("TD RemoteView is an opaque communication endpoint")


def remote(buffer, peer):
    """Select a remote owner without issuing a load or a transfer."""
    if not isinstance(buffer, tirx.Buffer):
        raise TypeError("TD.remote expects a symmetric Buffer before slicing")
    spec = _spec(buffer)
    peer = _int32(peer, "peer")
    if isinstance(peer, tirx.IntImm) and not 0 <= peer.value < spec.group.size:
        raise ValueError("TD.remote: peer is outside the group")
    return RemoteView(buffer, peer, spec)


def _endpoint(value, channel, *, access, signal=False):
    remote_endpoint = isinstance(value, (RemoteRegion, RemoteView))
    if isinstance(value, RemoteView):
        value = RemoteRegion(_region(value.buffer), value.peer, value.spec)
    region = value.region if remote_endpoint else _region(value)
    spec = value.spec if remote_endpoint else _spec(region.buffer)
    if spec.group != channel.group:
        raise ValueError("TD endpoint and channel must belong to the same group")
    if (spec.kind == "signal") != signal:
        raise TypeError(
            "TD signals may only be accessed with signal APIs" if spec.kind == "signal" else "TD signal API requires SignalBuffer"
        )
    if remote_endpoint and access not in spec.remote_access and spec.remote_access != "read_write":
        raise ValueError(f"TD remote {access} violates {spec.remote_access} permission for {region.buffer.name}")
    analyzer = tvm.arith.Analyzer()
    if len(region.region) != len(region.buffer.shape):
        raise ValueError("TD region must specify every buffer dimension")
    for axis, bound in zip(region.region, region.buffer.shape):
        if analyzer.can_prove(axis.extent <= 0):
            raise ValueError("TD regions must be nonempty")
        if analyzer.can_prove(axis.min < 0) or analyzer.can_prove(axis.min + axis.extent > bound):
            raise ValueError("TD region is out of bounds")
    if signal:
        if any(not analyzer.can_prove(axis.extent == 1) for axis in region.region):
            raise ValueError("TD signal API requires one signal cell")
    else:
        # A rectangular region is contiguous only when dimensions following
        # its first non-unit extent cover complete physical rows.
        partial = False
        for axis, bound in zip(region.region, region.buffer.shape):
            if partial and not (analyzer.can_prove(axis.min == 0) and analyzer.can_prove(axis.extent == bound)):
                raise ValueError("TD P0 requires contiguous regions; explicitly pack strided tiles")
            if not analyzer.can_prove(axis.extent == 1):
                partial = True
    return region, value.peer if remote_endpoint else tirx.IntImm("int32", -1), spec, remote_endpoint


def _annotations(channel, **extra):
    _channel(channel)
    return {"schema_version": 1, "group": channel.group.name, "group_size": channel.group.size, "channel": channel.name, **extra}


def _emit(op, args, annotations):
    annotations = {key: tirx.StringImm(value) if isinstance(value, str) else value for key, value in annotations.items()}
    ir_builder.evaluate(tirx.call_intrin("int32", tirx.op.Op.get(f"tl.distributed.{op}"), *args, annotations=annotations))


def rank(group):
    """Return the launch's logical rank as int32, never a device identifier."""
    _group(group)
    return tirx.call_intrin(
        "int32", tirx.op.Op.get("tl.distributed.rank"), group.name, annotations={"group_size": group.size, "schema_version": 1}
    )


def size(group):
    _group(group)
    return tirx.IntImm("int32", group.size)


@dataclass(frozen=True, eq=False)
class CommEvent:
    """Opaque completion identity, scoped to one structured issuing region."""

    event_id: int
    direction: str
    channel: ChannelSpec
    builder: object = field(repr=False)
    frames: tuple = field(repr=False)

    def __bool__(self):
        raise TypeError("TD.CommEvent is opaque; use TD.wait for completion")


def _event(direction, channel):
    builder = _builder()
    state = _channel(channel)
    event = CommEvent(state.event_count, direction, channel, builder, _event_frames(builder))
    state.event_count += 1
    return event


def _event_frames(builder):
    from tilelang.language.eager.builder import TIR_CONTROL_FRAME

    return tuple(id(frame) for frame in builder.frames if isinstance(frame, TIR_CONTROL_FRAME))


def _copy_endpoints(src, dst, channel):
    _channel(channel)
    source = _endpoint(src, channel, access="read")
    target = _endpoint(dst, channel, access="write")
    if source[3] == target[3]:
        raise ValueError("TD.copy_async requires exactly one remote endpoint; use T.copy for local copies")
    analyzer = tvm.arith.Analyzer()
    if source[0].buffer.dtype != target[0].buffer.dtype:
        raise ValueError("TD transfers require identical dtype; cast explicitly before communication")
    src_size = math.prod(axis.extent for axis in source[0].region)
    dst_size = math.prod(axis.extent for axis in target[0].region)
    if not analyzer.can_prove(src_size == dst_size):
        raise ValueError("TD transfers require equal element counts")
    return source, target


def copy_async(src, dst, *, channel):
    source, target = _copy_endpoints(src, dst, channel)
    event = _event("get" if source[3] else "put", channel)
    _emit(
        "copy_async",
        [event.event_id, source[1], target[1]],
        _annotations(channel, src_region=source[0], dst_region=target[0], direction=event.direction),
    )
    return event


def wait(event, *, completion):
    if not isinstance(event, CommEvent):
        raise TypeError("TD.wait requires the CommEvent returned by an issuing operation")
    builder = _builder()
    if event.builder is not builder or event.frames != _event_frames(builder):
        raise ValueError("TD.wait must occur in the issuing structured branch and loop iteration")
    if completion not in ("local", "remote"):
        raise ValueError("TD.wait completion must be local or remote")
    if completion == "remote" and event.direction == "get":
        raise ValueError("TD.get supports only local completion")
    _emit("wait", [event.event_id, completion], _annotations(event.channel, direction=event.direction))


def _signal_value(value):
    value = _int32(value, "signal value")
    if isinstance(value, tirx.IntImm) and value.value <= 0:
        raise ValueError("TD published signal values must be in 1..2^31-1; zero is initialization only")
    return value


def signal(dst, value, *, channel, order="release", release_regions=()):
    _channel(channel)
    if order != "release":
        raise ValueError("TD.signal supports only release ordering")
    destination = _endpoint(dst, channel, access="write", signal=True)
    if not destination[3]:
        raise ValueError("TD.signal requires a remote signal destination")
    regions = []
    for value_region in release_regions:
        endpoint = _endpoint(value_region, channel, access="read")
        if endpoint[3]:
            raise ValueError("TD.release_regions must be local regions")
        regions.append(endpoint[0])
    _emit(
        "signal",
        [destination[1], _signal_value(value)],
        _annotations(channel, signal_region=destination[0], release_regions=regions, order=order),
    )


def wait_signal(src, value, *, channel, cmp="ge", order="acquire", covers):
    _channel(channel)
    if cmp != "ge" or order != "acquire":
        raise ValueError("TD.wait_signal supports only cmp='ge', order='acquire'")
    source = _endpoint(src, channel, access="read", signal=True)
    if source[3]:
        raise ValueError("TD.wait_signal requires a local SignalBuffer cell")
    covered = [_endpoint(region, channel, access="read") for region in covers]
    _emit(
        "wait_signal",
        [_signal_value(value)],
        _annotations(
            channel, signal_region=source[0], covers=[x[0] for x in covered], covers_peers=[x[1] for x in covered], cmp=cmp, order=order
        ),
    )


def put_signal(src, dst, *, signal, value, channel):
    source, destination = _copy_endpoints(src, dst, channel)
    notification = _endpoint(signal, channel, access="write", signal=True)
    if source[3] or not destination[3] or not notification[3]:
        raise ValueError("TD.put_signal requires a local payload source and remote payload/signal destinations")
    if not tvm.arith.Analyzer().can_prove(destination[1] == notification[1]):
        raise ValueError("TD.put_signal payload and signal must target the same peer")
    event = _event("put", channel)
    _emit(
        "put_signal",
        [event.event_id, destination[1], _signal_value(value)],
        _annotations(channel, src_region=source[0], dst_region=destination[0], signal_region=notification[0], direction="put"),
    )
    return event


def fence(*, channel):
    _emit("fence", [], _annotations(channel))


def quiet(*, channel):
    _emit("quiet", [], _annotations(channel))


def normalize_prim_func(func, builder):
    """Append the ordinary Buffer descriptor ABI before parameter extraction."""
    state = builder.distributed_state
    if len(state.groups) != 1 or len(state.channels) > 1:
        raise ValueError("TD ascend950_cannsim_v1 supports one static group and at most one channel per kernel")
    signal_data = {data for data, (_, spec) in state.buffers.items() if spec.kind == "signal"}

    def check_signal_access(node):
        # Signal API operands have already become BufferRegion annotations;
        # an actual load/store or escaping pointer is ordinary data access.
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)) and node.buffer.data in signal_data:
            raise TypeError(f"TD SignalBuffer {node.buffer.name} only accepts signal APIs; ordinary load/store/copy is forbidden")
        if isinstance(node, tirx.Call) and any(isinstance(arg, tirx.Var) and arg in signal_data for arg in node.args):
            raise TypeError("TD SignalBuffer pointer escape is forbidden; use signal APIs")

    tirx.stmt_functor.post_order_visit(func.body, check_signal_access)
    objects = []
    buffer_specs = {}
    for index, param in enumerate(func.params):
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        item = state.buffers.get(buffer.data)
        if item is None:
            continue
        spec = item[1]
        entry = {
            "key": buffer.name,
            "param_index": index,
            "group": spec.group.name,
            "kind": spec.kind,
            "dtype": str(buffer.dtype),
            "shape": list(spec.shape),
            "strides": list(spec.strides),
            "remote_access": spec.remote_access,
            "alignment": 64,
            "physical_nbytes": spec.physical_nbytes,
        }
        objects.append(entry)
        buffer_specs[buffer.data] = entry
    descriptor_nbytes = 64 + sum(group.size for group in state.groups.values()) * len(objects) * 32
    descriptor = tirx.decl_buffer((descriptor_nbytes,), "uint8", name="__comm_descriptor", scope="global", data_alignment=64)
    params = [*func.params, descriptor.data]
    buffer_map = dict(func.buffer_map)
    buffer_map[descriptor.data] = descriptor
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "groups": [{"name": x.name, "size": x.size} for x in sorted(state.groups.values(), key=lambda x: x.name)],
        "channels": [{"name": x.name, "group": x.group.name} for x in sorted(state.channels.values(), key=lambda x: x.name)],
        "objects": objects,
        "descriptor": {
            "param_index": len(func.params),
            "parameter": descriptor.name,
            "nbytes": descriptor_nbytes,
            "abi_version": 1,
            "encoding": "cpu-model-v1",
        },
    }
    result = tirx.PrimFunc(params, func.body, ret_type=func.ret_type, buffer_map=buffer_map, attrs=func.attrs, span=func.span)
    if func.attrs is not None and "tilelang_out_idx" in func.attrs:
        # T.empty records returned buffers relative to the original argument
        # tail. Appending the private descriptor must not change those outputs.
        indices = [int(index) - 1 if int(index) < 0 else int(index) for index in func.attrs["tilelang_out_idx"]]
        result = result.with_attr("tilelang_out_idx", indices)
    return result.with_attr(MANIFEST_ATTR, json.dumps(manifest, sort_keys=True, separators=(",", ":"))).with_attr(
        BUFFER_SPECS_ATTR, buffer_specs
    )


__all__ = [
    "GroupSpec",
    "ChannelSpec",
    "SymBuffer",
    "SignalBuffer",
    "RemoteRegion",
    "RemoteView",
    "CommEvent",
    "rank",
    "size",
    "remote",
    "copy_async",
    "wait",
    "signal",
    "wait_signal",
    "put_signal",
    "fence",
    "quiet",
]
