"""Compile-only coverage for native SHMEM device calls and their launch contract."""

import os
from pathlib import Path

import pytest

from tilelang.ascend.distributed.transport import (
    adapt_cann91_shmem,
    lower_transport,
    make_transport_prim_func,
    shmem_include_flags,
    transport_pass_configs,
)


@pytest.mark.parametrize("kind", ["put", "get", "put_signal", "signal", "wait_signal", "fence", "quiet", "local_copy"])
def test_one_aiv_and_issuer_local_completion(kind):
    nbytes = 8192 if kind in {"put", "get", "put_signal", "local_copy"} else 0
    source = lower_transport(kind, nbytes)
    assert "__global__ __vector__" in source
    assert "__mix__" not in source
    assert "aclshmemx_mte_quiet();" in source
    if kind in {"put", "get", "put_signal", "local_copy"}:
        assert "__ubuf__ uint8_t *scratch" in source
        assert ", 4096, 8192, peer, 0)" in source
    if kind == "put_signal":
        put = source.index("aclshmemx_mte_put_nbi")
        drain = source.index("aclshmemx_mte_quiet", put)
        signal = source.index("aclshmemx_mte_atomic_set<uint32_t>")
        assert put < drain < signal < source.rindex("aclshmemx_mte_quiet")
    if kind == "wait_signal":
        assert "aclshmem_uint32_wait_until" in source
        assert "static_assert(ACLSHMEM_CMP_GE == 3" in source


def test_transport_abi_preserves_logical_argument_order():
    func = make_transport_prim_func("put_signal", 64)
    assert [str(p.name) for p in func.params] == ["Dst_handle", "Src_handle", "peer", "Signal_handle", "value"]
    assert [str(p.dtype) for p in func.params] == ["handle", "handle", "int32", "handle", "uint32"]
    assert tuple(int(i) for i in func.buffer_map[func.params[3]].shape) == (16,)


@pytest.mark.parametrize("nbytes", [0, -32, 1, 33, 2**32, True, 32.0])
def test_bad_mte_payload_rejected(nbytes):
    with pytest.raises(ValueError, match="(payload|integer)"):
        make_transport_prim_func("put", nbytes)


def test_non_payload_length_and_unknown_operation_rejected():
    with pytest.raises(ValueError, match="payload length"):
        make_transport_prim_func("signal", 32)
    with pytest.raises(ValueError, match="unsupported"):
        make_transport_prim_func("reduce", 32)


def test_compatibility_preserves_device_calls():
    source = lower_transport("put_signal", 64)
    adapted = adapt_cann91_shmem(source)
    assert '#include "kernel_operator.h"' in adapted
    assert adapted[adapted.index('extern "C"') :] == source[source.index('extern "C"') :]
    with pytest.raises(ValueError, match="pure Vector"):
        adapt_cann91_shmem(source.replace("__vector__", "__cube__"))
    with pytest.raises(ValueError, match="SDK operations"):
        adapt_cann91_shmem(source + "\nasc_unknown_instruction();\n")
    with pytest.raises(ValueError, match="unsupported"):
        transport_pass_configs(sdk_compatibility="unknown")


def test_transitive_header_change_invalidates_compile_flags(tmp_path):
    include = tmp_path / "include"
    device = tmp_path / "src/device/gm2gm/engine"
    include.mkdir()
    device.mkdir(parents=True)
    (include / "shmem.h").write_text("// API\n")
    implementation = device / "shmem_device_mte.hpp"
    implementation.write_text("// version 1\n")
    before = shmem_include_flags(tmp_path)
    implementation.write_text("// version 2\n")
    after = shmem_include_flags(tmp_path)
    assert before[:-1] == after[:-1]
    assert before[-1] != after[-1]
    implementation.unlink()
    with pytest.raises(ValueError, match="complete SHMEM"):
        shmem_include_flags(tmp_path)


@pytest.mark.parametrize("kind", ["put_signal", "get", "signal", "wait_signal", "quiet"])
def test_real_sdk_compiles_shmem_calls_without_npu(kind, tmp_path):
    root = os.environ.get("SHMEM_ROOT")
    if not root:
        pytest.skip("set SHMEM_ROOT to enable actual SHMEM header and Bisheng compilation")
    from tilelang.contrib import bisheng

    nbytes = 8192 if kind in {"put_signal", "get"} else 0
    source = adapt_cann91_shmem(lower_transport(kind, nbytes))
    output = tmp_path / f"{kind}.aibin"
    blob = bisheng.compile_ascend(
        source,
        target_format="aibin",
        npu_arch="dav-3510",
        options=shmem_include_flags(root),
        path_target=str(output),
    )
    assert bytes(blob).startswith(b"\x7fELF")
    assert Path(output).is_file()
