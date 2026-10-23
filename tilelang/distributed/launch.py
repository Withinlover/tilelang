"""Host binding for the local adapter; no alternative device JIT backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .abi import NATIVE_PROFILE, encode_descriptor, kernel_manifest, manifest_digest
from .session import LaunchTicket, Session, SymmetricBuffer, UnsupportedProfileError, _check_ticket, _validate_manifest


@dataclass(eq=False)
class Run:
    session: Session
    ticket: LaunchTicket
    result: object = None
    finished: bool = False
    completed: bool = False


class RunFailedError(RuntimeError):
    """A local adapter returned a failed report; ``run`` retains its evidence."""

    def __init__(self, run):
        super().__init__("distributed execution failed; inspect the adapter report")
        self.run = run


def launch(kernel, *, ticket, buffers=None, scalars=None, stream=None, **kwargs):
    ctx = ticket.session
    if ctx.profile == NATIVE_PROFILE:
        from tilelang.ascend.distributed.native import launch_native

        return launch_native(kernel, ticket=ticket, buffers=buffers, scalars=scalars, stream=stream, **kwargs)
    with ctx._lock:
        _check_ticket(ctx, ticket)
        if ctx.state != "PREPARED":
            raise RuntimeError(f"launch requires the current PREPARED ticket; found {ctx.state}")
        if kernel is not ticket.kernel or manifest_digest(kernel_manifest(kernel)) != ticket.manifest_digest:
            raise ValueError("launch kernel or its manifest differs from the prepared binding")
        if manifest_digest(ticket.manifest) != ticket.manifest_digest:
            raise ValueError("prepared manifest was modified")
        _validate_manifest(ctx, ticket.manifest, ticket.channel)
        if set(ticket.descriptors) != set(range(ctx.world_size)):
            raise ValueError("prepared descriptors do not cover every logical rank")
        for rank, descriptor in ticket.descriptors.items():
            expected = encode_descriptor(
                ticket.manifest,
                rank=rank,
                generation=ticket.generation,
                launch_id=ticket.launch_id,
                epoch_base=ticket.base,
                epoch_end=ticket.end,
            )
            if not isinstance(descriptor, np.ndarray) or descriptor.dtype != np.uint8 or not np.array_equal(descriptor, expected):
                raise ValueError("prepared CPU descriptor was modified")
        if stream is not None:
            raise UnsupportedProfileError("CPU co-simulation does not accept a device stream")
        bound_buffers = {spec["key"]: ctx.objects[spec["key"]] for spec in ticket.manifest["objects"]}
        if buffers is not None:
            for name, value in buffers.items():
                if name in bound_buffers and value is not bound_buffers[name]:
                    raise ValueError(f"symmetric binding {name!r} must be the prepared allocation")
                if isinstance(value, SymmetricBuffer) and value.session is not ctx:
                    raise ValueError(f"buffer {name!r} belongs to a different session")
                bound_buffers[name] = value
        bound_scalars = dict(scalars or {})
        if "epoch0" in bound_scalars and (isinstance(bound_scalars["epoch0"], bool) or bound_scalars["epoch0"] != ticket.base):
            raise ValueError("epoch0 must equal the collectively prepared ticket base")
        bound_scalars.setdefault("epoch0", ticket.base)
        if not callable(getattr(kernel, "run", None)):
            raise TypeError("the local profile requires a compiled distributed program with run()")
        run = Run(ctx, ticket)
        ctx._run, ctx.state = run, "RUNNING"
    try:
        # Release the lock while work runs; another caller sees RUNNING and fails
        # before touching the arrays instead of racing this launch.
        result = kernel.run(session=ctx, ticket=ticket, buffers=bound_buffers, scalars=bound_scalars, **kwargs)
        run.result = result
        if (isinstance(result, dict) and result.get("passed") is False) or getattr(result, "passed", True) is False:
            raise RunFailedError(run)
        run.finished = True
    except BaseException:
        with ctx._lock:
            ctx.state = "FAILED"
        raise
    return run


def complete(run):
    ctx = run.session
    if ctx.profile == NATIVE_PROFILE:
        from tilelang.ascend.distributed.native import complete_native

        return complete_native(run)
    with ctx._lock:
        if run.completed:
            return run.result
        _check_ticket(ctx, run.ticket)
        if ctx._run is not run or ctx.state != "RUNNING" or not run.finished:
            raise RuntimeError("complete requires the successfully returned current run")
        run.completed, ctx.state = True, "LOCAL_DONE"
        return run.result
