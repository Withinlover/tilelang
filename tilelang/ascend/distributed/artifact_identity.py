"""Eligibility for checking a sealed source by its immutable root identity.

Unknown metadata is legal: False asks the artifact path to retain its original
full save_json comparison. In particular, a readonly DictAttrs field can still
contain mutable ffi.Dict/ffi.List objects, even after a save/load round trip.
"""

import json

from tvm import DataType, tirx
from tvm_ffi.container import Array, Map


# This is an exact audit list, deliberately not a namespace-prefix allowlist.
# TIR fields (including Call.annotations, Buffer, and PrimFunc's inherited
# attrs/struct_info_) are registered def_ro in include/tvm/tirx/{expr,stmt,
# buffer,function}.h and include/tvm/ir/{expr,function}.h. The structural types
# below use def_ro in include/tvm/{ir/type.h,relax/{expr,struct_info}.h}.
# DictAttrs stores an immutable Map (include/tvm/ir/attrs.h); its contents still
# require recursive checking. SourceName/Span use def_ro in ir/source_map.h.
# AscendFractalLayout's kind and its Layout parent's input_size/forward_index
# are def_ro in src/ascend/layout/ascend_layouts.cc and src/layout/layout.cc.
# Array/Map are immutable Sequence/Mapping in tvm_ffi/container.py; mutable
# List/Dict, Tensor storage, IRModule, SourceMap and unknown objects are absent.
_READONLY_TYPES = frozenset(
    {
        "ffi.Array",
        "ffi.Map",
        "ffi.String",
        "ir.DictAttrs",
        "ir.IntImm",
        "ir.FloatImm",
        "ir.Op",
        "ir.PointerType",
        "ir.PrimType",
        "ir.Range",
        "ir.SourceName",
        "ir.Span",
        "ir.TupleType",
        "relax.FuncStructInfo",
        "relax.PrimStructInfo",
        "relax.ShapeStructInfo",
        "relax.TensorStructInfo",
        "relax.TupleStructInfo",
        "relax.expr.ShapeExpr",
        "tirx.Add",
        "tirx.And",
        "tirx.AttrStmt",
        "tirx.Bind",
        "tirx.Buffer",
        "tirx.BufferLoad",
        "tirx.BufferRegion",
        "tirx.BufferStore",
        "tirx.Call",
        "tirx.Cast",
        "tirx.Div",
        "tirx.EQ",
        "tirx.Evaluate",
        "tirx.FloorDiv",
        "tirx.FloorMod",
        "tirx.For",
        "tirx.GE",
        "tirx.GT",
        "tirx.IfThenElse",
        "tirx.IterVar",
        "tirx.LE",
        "tirx.LT",
        "tirx.Max",
        "tirx.Min",
        "tirx.Mod",
        "tirx.Mul",
        "tirx.NE",
        "tirx.Not",
        "tirx.Or",
        "tirx.PrimFunc",
        "tirx.SBlock",
        "tirx.SBlockRealize",
        "tirx.Select",
        "tirx.SeqStmt",
        "tirx.StringImm",
        "tirx.Sub",
        "tirx.Var",
        "tl.AscendFractalLayout",
    }
)
_JSON_SCALARS = frozenset({"None", "bool", "int", "float", "str", "DataType"})


def source_supports_identity_guard(program, ir_json):
    """Return whether a validated snapshot needs only a root/rebind guard.

    Run once at snapshot creation, not on every launch. The caller must still
    reject a replaced root, changed manifest/compile options and replaced kernel
    handles. This proves stability of the serialized source under supported
    public APIs; it does not police unsafe native memory writes or global Op
    registry changes (which the old save_json comparison also cannot observe).
    """
    if not isinstance(program, tirx.PrimFunc):
        return False
    try:
        nodes = json.loads(ir_json)["nodes"]
    except (TypeError, ValueError, KeyError):
        return False
    if not isinstance(nodes, list) or not nodes:
        return False
    if any(not isinstance(node, dict) or node.get("type") not in _READONLY_TYPES | _JSON_SCALARS for node in nodes):
        return False
    # Check reflection flags as well as the audited type list so an SDK/FFI
    # change from def_ro to def_rw conservatively disables the optimization.
    # Retain wrappers to avoid Python-id reuse while recursively inspecting
    # containers and all inherited fields (not only ordinary TIR children).
    seen = {}

    def readonly(value):
        if value is None or isinstance(value, (str, bool, int, float, DataType)):
            return True
        if id(value) in seen:
            return True
        seen[id(value)] = value
        info = getattr(value, "__tvm_ffi_type_info__", None)
        if info is None or info.type_key not in _READONLY_TYPES:
            return False
        if isinstance(value, Array):
            return all(readonly(item) for item in value)
        if isinstance(value, Map):
            return all(readonly(key) and readonly(item) for key, item in value.items())
        if info.type_key == "ir.Op":
            # Op's serialized identity is its registered name, not its global
            # attribute registry. Match the existing JSON check's boundary.
            return True
        while info is not None:
            for field in info.fields:
                if field.frozen is not True or not readonly(field.getter(value)):
                    return False
            info = info.parent_type_info
        return True

    try:
        return readonly(program)
    except (AttributeError, TypeError, ValueError):
        # Reflection is an optional proof mechanism. A changed/missing Python
        # reflection API is inconclusive, not a new source-legality failure.
        return False
