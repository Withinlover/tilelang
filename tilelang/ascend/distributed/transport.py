"""Single-issuer SHMEM MTE kernels for the Ascend 950 sequential profile.

These kernels use the existing Ascend pipeline and ``tvm_ffi`` executor. All
external calls name public SHMEM functions; no private SHMEM state layout or
second kernel launcher is embedded here. Each kernel drains its own MTE work
before returning, so a later kernel may safely run on another physical AIV.

Visibility contract (pinned SHMEM revision below): its public shmem_device_mo.h
guarantees that completed symmetric writes are visible to the destination PE.
For MTE-only initialization, shmem_device_mo.hpp implements that completion as
PIPE_ALL plus whole scalar DCache clean/invalidate, exactly as mte_quiet does.
The signal wait invalidates its scalar cache line on each observation. Payload
consumers in this profile use ordinary MTE2 GM-to-L1 loads; they do not use
Scalar, SIMT or NDDMA cached loads. Ascend 950 L2 is shared between local cores;
flushing an AIV scalar cache is not presented as an L2 flush for another core.
The phased executor must order and complete earlier local compute on its stream
before launching a release signal, and complete the wait before later compute.
Hot-slot reuse is permitted only after the receiving GEMM completes and ACK is
observed. Hardware validation must exercise changing data across those epochs.
"""

import hashlib
import json
from pathlib import Path
import re
import shlex

import tilelang
import tilelang.ascend.language as T


SHMEM_SOURCE_REVISION = "a232a76516c435cd39ecd10125fc6e14a55fc18b"
SHMEM_SOURCE_URL = "https://gitcode.com/cann/shmem.git"
TRANSPORT_KINDS = frozenset({"put", "get", "put_signal", "signal", "wait_signal", "fence", "quiet", "local_copy"})
SCRATCH_BYTES = 4096
SIGNAL_WORDS = 16
SHMEM_IMPORT = '#include "shmem.h"\nstatic_assert(ACLSHMEM_CMP_GE == 3, "SHMEM compare ABI changed");\n'


def make_transport_prim_func(kind: str, nbytes: int = 0):
    """Create a TD-free, one-AIV communication kernel.

    Payload arguments are flat uint8 tensors. Signal arguments are one 64-byte
    uint32 view of the logical int32 signal cell; nonnegative DSL epochs have
    identical representations. Arguments, in order, are:

    * put/get/local_copy: ``dst, src, peer`` (local_copy requires the local rank);
    * put_signal: ``dst, src, peer, signal, value``;
    * signal: ``signal, peer, value``;
    * wait_signal: ``signal, value``;
    * fence/quiet: no arguments.

    The remote pointer is always the corresponding *local symmetric* address;
    the public SHMEM API translates it using ``peer``. Payload transfers must
    cover an integral number of 32-byte MTE blocks. The higher-level profile
    checks regions and allocation provenance before creating these views.
    """
    if kind not in TRANSPORT_KINDS:
        raise ValueError(f"unsupported SHMEM transport operation: {kind}")
    payload = kind in {"put", "get", "put_signal", "local_copy"}
    if not isinstance(nbytes, int) or isinstance(nbytes, bool):
        raise ValueError("SHMEM transport nbytes must be an integer")
    if payload and (nbytes <= 0 or nbytes % 32 or nbytes > 2**32 - 1):
        raise ValueError("SHMEM MTE payload must be a positive multiple of 32 bytes below 2**32")
    if not payload and nbytes:
        raise ValueError("signal and drain operations do not take a payload length")

    if kind in {"put", "get", "local_copy"}:
        operation = "aclshmemx_mte_get_nbi<uint8_t>" if kind == "get" else "aclshmemx_mte_put_nbi<uint8_t>"

        @T.prim_func
        def transfer(Dst: T.Tensor((nbytes,), "uint8"), Src: T.Tensor((nbytes,), "uint8"), peer: T.int32):
            with T.Kernel(1):
                T.import_source(SHMEM_IMPORT)
                scratch = T.alloc_shared((SCRATCH_BYTES,), "uint8")
                T.evaluate(
                    T.call_extern(
                        "int32",
                        operation,
                        T.access_ptr(Dst[0], "w", extent=nbytes),
                        T.access_ptr(Src[0], "r", extent=nbytes),
                        T.access_ptr(scratch[0], "rw", extent=SCRATCH_BYTES),
                        SCRATCH_BYTES,
                        nbytes,
                        peer,
                        0,
                    )
                )
                T.evaluate(T.call_extern("int32", "aclshmemx_mte_quiet"))

        return transfer.with_attr("global_symbol", f"td_shmem_{kind}_{nbytes}")

    if kind == "put_signal":

        @T.prim_func
        def put_signal(
            Dst: T.Tensor((nbytes,), "uint8"),
            Src: T.Tensor((nbytes,), "uint8"),
            peer: T.int32,
            Signal: T.Tensor((SIGNAL_WORDS,), "uint32"),
            value: T.uint32,
        ):
            with T.Kernel(1):
                T.import_source(SHMEM_IMPORT)
                scratch = T.alloc_shared((SCRATCH_BYTES,), "uint8")
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "aclshmemx_mte_put_nbi<uint8_t>",
                        T.access_ptr(Dst[0], "w", extent=nbytes),
                        T.access_ptr(Src[0], "r", extent=nbytes),
                        T.access_ptr(scratch[0], "rw", extent=SCRATCH_BYTES),
                        SCRATCH_BYTES,
                        nbytes,
                        peer,
                        0,
                    )
                )
                # Complete the payload on this issuer before publishing READY.
                T.evaluate(T.call_extern("int32", "aclshmemx_mte_quiet"))
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "aclshmemx_mte_atomic_set<uint32_t>",
                        T.access_ptr(Signal[0], "w", extent=1),
                        value,
                        peer,
                    )
                )
                T.evaluate(T.call_extern("int32", "aclshmemx_mte_quiet"))

        return put_signal.with_attr("global_symbol", f"td_shmem_put_signal_{nbytes}")

    if kind == "signal":

        @T.prim_func
        def signal(Signal: T.Tensor((SIGNAL_WORDS,), "uint32"), peer: T.int32, value: T.uint32):
            with T.Kernel(1):
                T.import_source(SHMEM_IMPORT)
                T.evaluate(T.call_extern("int32", "aclshmemx_mte_quiet"))
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "aclshmemx_mte_atomic_set<uint32_t>",
                        T.access_ptr(Signal[0], "w", extent=1),
                        value,
                        peer,
                    )
                )
                T.evaluate(T.call_extern("int32", "aclshmemx_mte_quiet"))

        return signal.with_attr("global_symbol", "td_shmem_signal")

    if kind == "wait_signal":

        @T.prim_func
        def wait_signal(Signal: T.Tensor((SIGNAL_WORDS,), "uint32"), value: T.uint32):
            with T.Kernel(1):
                T.import_source(SHMEM_IMPORT)
                # SHMEM_IMPORT verifies the public comparison enum at build time.
                T.evaluate(
                    T.call_extern(
                        "int32",
                        "aclshmem_uint32_wait_until",
                        T.access_ptr(Signal[0], "r", extent=1),
                        3,
                        value,
                    )
                )
                T.evaluate(T.call_extern("int32", "aclshmemx_mte_quiet"))

        return wait_signal.with_attr("global_symbol", "td_shmem_wait_signal")

    @T.prim_func
    def drain():
        with T.Kernel(1):
            T.import_source(SHMEM_IMPORT)
            T.evaluate(T.call_extern("int32", "aclshmemx_mte_quiet"))

    return drain.with_attr("global_symbol", f"td_shmem_{kind}")


def shmem_include_flags(shmem_root):
    """Resolve installed/source SHMEM headers and bind their contents to JIT keys."""
    root = Path(shmem_root).expanduser().resolve()
    required = [root / "include/shmem.h", root / "src/device/gm2gm/engine/shmem_device_mte.hpp"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("SHMEM_ROOT must contain complete SHMEM include/ and src/device/ headers: " + ", ".join(missing))
    paths = [root / "include", root / "src", root / "src/device", root / "src/device_simt"]
    digest = hashlib.sha256()
    # include/ and src/ contain every selected header. Avoid walking unrelated
    # build/install trees, while preserving the exact path/content digest.
    for path in sorted(path for tree in paths[:2] for path in tree.rglob("*")):
        if path.is_file() and path.suffix in {".h", ".hpp"}:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    # The backend binary cache includes compiler options, not transitive header
    # bytes. This macro also invalidates shared JIT caches after an SDK edit.
    return [*(shlex.quote("-I" + str(path)) for path in paths), "-DTL_SHMEM_HEADER_DIGEST_" + digest.hexdigest() + "=1"]


def transport_pass_configs(shmem_root=None, *, sdk_compatibility="native"):
    config = {"tl.enable_auto_schedule": False}
    if shmem_root is not None:
        config["tl.device_compile_flags"] = shmem_include_flags(shmem_root)
    if sdk_compatibility != "native":
        if sdk_compatibility != "cann-9.1-shmem":
            raise ValueError(f"unsupported SHMEM SDK compatibility profile: {sdk_compatibility}")
        config["tl.ascend_sdk_compatibility"] = sdk_compatibility
    return config


def lower_transport(kind, nbytes=0, *, shmem_root=None):
    """Generate source without loading SHMEM, compiling device code or probing NPU."""
    func = make_transport_prim_func(kind, nbytes)
    with tilelang.transform.PassContext(config=transport_pass_configs()):
        return tilelang.lower(func, target={"kind": "ascend", "arch": "dav-3510"}).kernel_source


def build_transport(kind, nbytes=0, *, shmem_root, output_dir=None, sdk_compatibility="native"):
    """Compile through the shared tvm_ffi executor; do not create a new launcher."""
    func = make_transport_prim_func(kind, nbytes)
    config = transport_pass_configs(shmem_root, sdk_compatibility=sdk_compatibility)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        source = lower_transport(kind, nbytes)
        (out / f"{kind}_{nbytes}.asc").write_text(source)
        (out / f"{kind}_{nbytes}.tir").write_text(func.script(show_meta=True))
        (out / f"{kind}_{nbytes}.json").write_text(json.dumps({"kind": kind, "nbytes": nbytes, "pass_configs": config}, indent=2))
    return tilelang.compile(func, target={"kind": "ascend", "arch": "dav-3510"}, execution_backend="tvm_ffi", pass_configs=config)


def adapt_cann91_shmem(source):
    """Use CANN 9.1 headers for the audited single-AIV external SHMEM surface."""
    preamble = "#include <tl_templates/ascend/common.h>\n#include <tl_templates/ascend/debug.h>\n"
    if source.count(preamble) != 1 or "__global__ __vector__" not in source or "__mix__" in source:
        raise ValueError("cann-9.1-shmem requires the standard pure Vector source preamble")
    allowed = {"asc_init"}
    unexpected = set(re.findall(r"\b(asc_\w+)\s*\(", source)) - allowed
    if unexpected or '#include "shmem.h"' not in source:
        raise ValueError(f"cann-9.1-shmem does not translate these SDK operations: {sorted(unexpected)}")
    # Including kernel_operator before the low-level C API avoids conflicting
    # block-index macros in CANN 9.1's AscendC debug implementation.
    return source.replace(preamble, '#include "kernel_operator.h"\n#include <c_api/asc_simd.h>\n')
