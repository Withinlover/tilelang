"""P0 distributed source programs with explicit local and native profiles.

The native Ascend 950 profile uses SHMEM device communication and sequential
kernel phases. Physical two-card validation is separate from local simulation.
"""

from .abi import LOCAL_PROFILE, NATIVE_PROFILE

from .launch import Run, RunFailedError, complete, launch
from .session import (
    Channel,
    LaunchTicket,
    Session,
    SymmetricBuffer,
    UnsupportedProfileError,
    alloc_signals,
    alloc_symmetric,
    channel,
    close,
    copy_in,
    init,
    prepare_launch,
)
from .jit import BoundDistributedKernel, DistributedJIT, DistributedKernel, jit


def compile(program, *, profile=LOCAL_PROFILE, **kwargs):
    """Compile a canonical TD PrimFunc for an explicitly selected profile."""
    if profile == NATIVE_PROFILE:
        from tilelang.ascend.distributed.native import compile_program

        return compile_program(program, profile=profile, **kwargs)
    from tilelang.ascend.distributed.cosim import compile_program

    return compile_program(program, profile=profile, **kwargs)


__all__ = [
    "BoundDistributedKernel",
    "Channel",
    "DistributedJIT",
    "DistributedKernel",
    "LaunchTicket",
    "LOCAL_PROFILE",
    "NATIVE_PROFILE",
    "Run",
    "RunFailedError",
    "Session",
    "SymmetricBuffer",
    "UnsupportedProfileError",
    "alloc_signals",
    "alloc_symmetric",
    "channel",
    "close",
    "compile",
    "complete",
    "copy_in",
    "init",
    "jit",
    "launch",
    "prepare_launch",
]
