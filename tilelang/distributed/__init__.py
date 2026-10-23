"""Local P0 distributed semantics for two logical Ascend 950 ranks.

Only the CPU-coordinated CANN co-simulation profile is implemented. Real
multi-process groups and hardware transport profiles are rejected explicitly.
"""

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


def compile(program, **kwargs):
    """Compile the canonical TD PrimFunc for the local co-simulation adapter."""
    from tilelang.ascend.distributed.cosim import compile_program

    return compile_program(program, **kwargs)


__all__ = [
    "Channel",
    "LaunchTicket",
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
    "launch",
    "prepare_launch",
]
