"""Immutable-root eligibility is conservative; content checks remain fallback."""

import json

import pytest
import tvm_ffi
from tvm import ir, tirx

from tilelang import tvm
from tilelang.ascend.distributed.artifact_identity import source_supports_identity_guard


def source_fixture():
    buffer = tirx.decl_buffer((8,), "float32", name="source")
    return tirx.PrimFunc([buffer.data], tirx.Evaluate(buffer[0]), buffer_map={buffer.data: buffer}).with_attr(
        "global_symbol", "identity_fixture"
    )


def test_known_immutable_source_survives_snapshot_serialization():
    program = source_fixture()
    encoded = tvm.ir.save_json(program)
    snapshot = tvm.ir.load_json(encoded)
    assert source_supports_identity_guard(snapshot, encoded)


@pytest.mark.parametrize("mutable_kind", ["dict", "list"])
@pytest.mark.parametrize("location", ["function_attribute", "call_annotation"])
def test_mutable_metadata_survives_load_json_and_requires_content_checks(mutable_kind, location):
    mutable = tvm_ffi.Dict({"value": 1}) if mutable_kind == "dict" else tvm_ffi.List([1])
    if location == "function_attribute":
        program = source_fixture().with_attr("review", {"nested": [mutable]})
    else:
        call = tirx.Call("int32", ir.Op.get("tirx.exp"), [tirx.const(1, "float32")], annotations={"review": {"nested": [mutable]}})
        program = tirx.PrimFunc([], tirx.Evaluate(call))
    encoded = tvm.ir.save_json(program)
    snapshot = tvm.ir.load_json(encoded)
    assert not source_supports_identity_guard(snapshot, encoded)
    attrs = snapshot.attrs if location == "function_attribute" else snapshot.body.value.annotations
    changed = attrs["review"]["nested"][0]
    changed["value" if mutable_kind == "dict" else 0] = 2
    # Root identity did not change. The existing complete JSON check detects
    # the public, supported mutation; save/load is not an immutability proof.
    assert tvm.ir.save_json(snapshot) != encoded
    assert not source_supports_identity_guard(snapshot, encoded)


def test_unknown_object_metadata_uses_content_check_fallback():
    program = source_fixture().with_attr("module_metadata", tvm.IRModule({}))
    encoded = tvm.ir.save_json(program)
    assert not source_supports_identity_guard(program, encoded)


def test_type_allowlist_is_exact_and_does_not_accept_unknown_namespace_members():
    program = source_fixture()
    encoded = json.loads(tvm.ir.save_json(program))
    encoded["nodes"].append({"type": "tirx.FutureMutableMetadata", "data": {}})
    assert not source_supports_identity_guard(program, json.dumps(encoded))


def test_changed_reflection_writability_disables_the_fast_guard(monkeypatch):
    program = source_fixture()
    encoded = tvm.ir.save_json(program)
    # Check inherited fields too: attrs is a BaseFunc field, not PrimFunc's.
    field = next(item for item in program.__tvm_ffi_type_info__.parent_type_info.fields if item.name == "attrs")
    assert field.frozen
    monkeypatch.setattr(field, "frozen", False)
    assert not source_supports_identity_guard(program, encoded)


@pytest.mark.parametrize("failure", ["missing_getter", "invalid_getter", "missing_parent"])
def test_unavailable_reflection_falls_back_instead_of_rejecting_source(monkeypatch, failure):
    program = source_fixture()
    encoded = tvm.ir.save_json(program)
    info = program.__tvm_ffi_type_info__
    if failure == "missing_parent":
        monkeypatch.setattr(info, "parent_type_info", object())
    elif failure == "missing_getter":
        monkeypatch.setattr(info.fields[0], "getter", None)
    else:

        def invalid_getter(_):
            raise ValueError("reflection unavailable")

        monkeypatch.setattr(info.fields[0], "getter", invalid_getter)
    assert not source_supports_identity_guard(program, encoded)


def test_public_ir_updates_produce_new_roots_and_original_contents_stay_fixed():
    original = source_fixture()
    encoded = tvm.ir.save_json(original)
    changed_body = original.with_body(tirx.Evaluate(0))
    changed_attrs = original.with_attr("review", "new")
    assert not changed_body.same_as(original)
    assert not changed_attrs.same_as(original)
    assert tvm.ir.save_json(original) == encoded
    with pytest.raises(AttributeError):
        original.body = tirx.Evaluate(0)
    with pytest.raises((TypeError, AttributeError)):
        original.params[0] = original.params[0]
    with pytest.raises((TypeError, AttributeError)):
        original.buffer_map[original.params[0]] = next(iter(original.buffer_map.values()))
    buffer = next(iter(original.buffer_map.values()))
    with pytest.raises(AttributeError):
        buffer.data = buffer.data
    assert tvm.ir.save_json(original) == encoded


@pytest.mark.parametrize("encoded", ["invalid", "[]", "{}", '{"nodes":[]}', '{"nodes":[null]}'])
def test_unrecognized_serialization_format_falls_back(encoded):
    assert not source_supports_identity_guard(source_fixture(), encoded)
