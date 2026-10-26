"""Lazy distributed operators backed by a long-lived symmetric-heap Session.

This layer binds the existing native phased compiler/runtime. It does not add
an execution backend or change the ordinary JIT's distributed-IR gate.
"""

from __future__ import annotations

from copy import deepcopy
from functools import update_wrapper
import inspect
import math
from pathlib import Path
from threading import RLock

import numpy as np

from .abi import MAX_EPOCH, NATIVE_PROFILE, kernel_manifest, manifest_digest
from .launch import complete, launch
from .session import UnsupportedProfileError, _positive_int, _storage_dtype


def _names(value, role):
    if isinstance(value, str):
        raise TypeError(f"{role} must be a sequence of buffer names, not a string")
    names = tuple(value)
    if any(not isinstance(name, str) or not name.isidentifier() for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{role} must contain unique buffer names")
    if role == "inputs" and "scalars" in names:
        raise ValueError("the input name 'scalars' is reserved for runtime scalar arguments")
    return names


def _static_key(value):
    """Keep Python types in specialization keys, including bool versus int."""
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return kind.__name__, value
    if kind is float and math.isfinite(value):
        return "float", value.hex()
    if kind in (tuple, list):
        return kind.__name__, tuple(_static_key(item) for item in value)
    if kind is dict and all(isinstance(key, str) for key in value):
        return "dict", tuple((key, _static_key(item)) for key, item in sorted(value.items()))
    raise TypeError("distributed JIT static arguments must be finite Python scalars, lists, tuples or string-keyed dictionaries")


def _collective_result(session, stage, metadata, operation):
    """Exchange local failures before a peer advances to a device operation."""
    result, error = None, None
    try:
        result = operation()
    except Exception as caught:
        error = caught
    try:
        session.preflight(
            {
                "jit_stage": stage,
                **metadata,
                "error": None if error is None else f"{type(error).__name__}: {error}",
            }
        )
    except Exception as collective_error:
        if error is not None:
            raise collective_error from error
        raise
    if error is not None:
        raise error
    return result


class DistributedJIT:
    """Specialize a PrimFunc factory by its normalized static arguments."""

    def __init__(self, factory, *, inputs, outputs, epochs, compile_options):
        if not callable(factory):
            raise TypeError("D.jit expects a factory returning a distributed PrimFunc")
        self.factory = factory
        self.inputs, self.outputs = _names(inputs, "inputs"), _names(outputs, "outputs")
        self.signature = inspect.signature(factory)
        if isinstance(epochs, str):
            if epochs not in self.signature.parameters:
                raise ValueError("epochs must name a static factory parameter")
        else:
            _positive_int(epochs, "epochs")
        self.epochs = epochs
        self._compile_options = deepcopy(compile_options)
        self._cache = {}
        self._lock = RLock()
        update_wrapper(self, factory)

    def __call__(self, *args, **kwargs):
        arguments = self.signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        key = _static_key(dict(arguments.arguments))
        with self._lock:
            if key not in self._cache:
                epochs = arguments.arguments[self.epochs] if isinstance(self.epochs, str) else self.epochs
                _positive_int(epochs, "epochs")
                self._cache[key] = DistributedKernel(
                    self.factory(*arguments.args, **arguments.kwargs),
                    inputs=self.inputs,
                    outputs=self.outputs,
                    epochs=epochs,
                    compile_options=self._compile_options,
                )
            return self._cache[key]


class DistributedKernel:
    """An address-free specialization; bind it collectively to allocate storage."""

    def __init__(self, prim_func, *, inputs, outputs, epochs, compile_options):
        from tilelang import tvm

        if not isinstance(prim_func, tvm.tirx.PrimFunc):
            raise TypeError("a D.jit factory must return a distributed PrimFunc")
        self._prim_func = prim_func
        self.inputs, self.outputs, self.epochs = inputs, outputs, epochs
        self._compile_options = deepcopy(compile_options)
        self._programs = {}
        self._lock = RLock()
        manifest = kernel_manifest(prim_func)
        objects = {obj["key"]: obj for obj in manifest["objects"]}
        for name in (*inputs, *outputs):
            if name not in objects or objects[name]["kind"] != "symmetric":
                raise ValueError(f"JIT input/output {name!r} must name a declared symmetric data buffer")
        self._contract = {"inputs": list(inputs), "outputs": list(outputs), "epochs": epochs}

    @property
    def prim_func(self):
        """The source of this immutable specialization, available for inspection."""
        return self._prim_func

    def bind(self, session, *, key=None):
        """Collectively allocate or reuse this operator's private heap blocks.

        Every rank must bind and call operators in the same order. The Session
        owns allocations until close; dropping this Python object does not free
        memory or resize the fixed symmetric heap.
        """
        from tilelang.ascend.distributed.native import NativeCompiledProgram
        from tilelang.ascend.distributed.native_runtime import NativeSession

        if not isinstance(session, NativeSession) or session.profile != NATIVE_PROFILE:
            raise UnsupportedProfileError("D.jit.bind requires a native SHMEM Session")
        with session._lock:
            if session.state not in ("QUIESCENT", "LOCAL_DONE"):
                raise RuntimeError(f"JIT binding requires a quiescent or completed Session; found {session.state}")

            def compile_program():
                if key is not None and (not isinstance(key, str) or not key):
                    raise ValueError("binding key must be a nonempty string")
                options = dict(self._compile_options)
                root = options.get("shmem_root") or session.shmem_root
                if root is None:
                    raise ValueError("D.jit requires the Session's SHMEM source root")
                root = str(Path(root).expanduser().resolve())
                options["shmem_root"] = root
                with self._lock:
                    if root not in self._programs:
                        self._programs[root] = NativeCompiledProgram(self.prim_func, **options)
                    return self._programs[root]

            compiled = _collective_result(session, "compile", self._contract, compile_program)
            identity = {"manifest": manifest_digest(compiled.manifest), **self._contract}
            session.preflight({"jit_bind": identity, "key": key})
            binding = session.bind(compiled, key=key if key is not None else "jit/" + manifest_digest(identity))
            return BoundDistributedKernel(binding, inputs=self.inputs, outputs=self.outputs, epochs=self.epochs)


class BoundDistributedKernel:
    """Synchronous NumPy callable using cached, Session-owned workspace.

    BF16 inputs use raw uint16 storage, matching the existing native API. Each
    returned array owns its storage, so future calls and Session.close cannot
    overwrite it. Inputs listed in outputs are supported as explicit in/out
    buffers. Other data buffers are persistent private workspace.
    """

    def __init__(self, binding, *, inputs, outputs, epochs):
        self.binding, self.compiled = binding, binding.kernel
        self.inputs, self.outputs, self.epochs = inputs, outputs, epochs
        self.last_run = None
        self._signature = inspect.Signature([inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in inputs])

    def __call__(self, *args, scalars=None, **kwargs):
        session = self.binding.session
        with session._lock:
            if session.state not in ("QUIESCENT", "LOCAL_DONE"):
                raise RuntimeError(f"JIT call requires a quiescent or completed Session; found {session.state}")
            metadata = {"binding": self.binding.key, "epochs": self.epochs, "inputs": list(self.inputs), "outputs": list(self.outputs)}

            def validate():
                session._check_binding(self.binding)
                inputs = self._signature.bind(*args, **kwargs).arguments
                arrays = {}
                for name, value in inputs.items():
                    obj = self.binding.objects[name]
                    if not isinstance(value, np.ndarray):
                        raise TypeError(f"input {name!r} must be a NumPy array; NPU Tensor inputs are not supported yet")
                    if value.shape != obj.shape or value.dtype != _storage_dtype(obj.dtype):
                        raise ValueError(f"input {name!r} expects {obj.shape} / {_storage_dtype(obj.dtype)}")
                    arrays[name] = np.ascontiguousarray(value)
                values = {} if scalars is None else dict(scalars)
                if "epoch0" in values:
                    raise ValueError("epoch0 is managed by the distributed JIT")
                if any(not isinstance(name, str) for name in values):
                    raise TypeError("runtime scalar names must be strings")
                for name, value in values.items():
                    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                        raise TypeError(f"runtime scalar {name!r} must be an integer")
                    values[name] = int(value)
                if session._epoch + self.epochs > MAX_EPOCH:
                    raise OverflowError("int32 signal epochs exhausted; close and recreate the Session")
                self.compiled.plan(
                    scalars=values,
                    epoch_base=session._epoch,
                    epoch_end=session._epoch + self.epochs,
                    initial_signals=self.binding.signal_values,
                )
                return arrays, values

            arrays, values = _collective_result(session, "validate", metadata, validate)
            # Scalars select the same finite host plan on every rank. Array
            # contents are deliberately not exchanged: ranks own different data.
            session.preflight({"jit_arguments": metadata, "scalars": values})
            ticket = session.prepare(
                self.compiled,
                channel=self.binding.channel,
                epochs=self.epochs,
                protocol="distributed-jit-v1",
                parameters={"binding": self.binding.key, "scalars": values},
                binding=self.binding,
            )
            try:

                def copy_inputs():
                    for name, array in arrays.items():
                        session.copy_in(self.binding.objects[name], array, ticket=ticket)

                _collective_result(session, "copy_in", metadata, copy_inputs)
                run = launch(self.compiled, ticket=ticket, scalars=values)
                self.last_run = run
                complete(run)
                results = _collective_result(
                    session,
                    "copy_out",
                    metadata,
                    lambda: tuple(session.copy_out(self.binding.objects[name]) for name in self.outputs),
                )
            except BaseException as error:
                session.fail(error)
                raise
            if not results:
                return None
            return results[0] if len(results) == 1 else results


def jit(func=None, *, inputs, outputs, epochs, profile=NATIVE_PROFILE, **compile_options):
    """Decorate a factory returning a canonical distributed PrimFunc.

    ``epochs`` is a positive integer or the name of a static factory parameter.
    Calling the decorated factory specializes it without allocating device
    memory. Call ``.bind(session)`` collectively to allocate persistent private
    workspace, then invoke the bound operator with NumPy inputs. Device kernels
    compile lazily through the existing native phased/tvm_ffi path. Pass
    ``debug=True`` to retain source, build and per-launch diagnostics.
    """
    if profile != NATIVE_PROFILE:
        raise UnsupportedProfileError("D.jit currently supports only the native SHMEM phased profile")
    options = {"profile": profile, "debug": False, **compile_options}

    def decorate(factory):
        return DistributedJIT(factory, inputs=inputs, outputs=outputs, epochs=epochs, compile_options=options)

    return decorate if func is None else decorate(func)
