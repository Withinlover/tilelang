"""Explicit capabilities; a device name alone never certifies a transport."""

from tilelang.analysis.distributed_checker import fail, reject_native_distributed  # noqa: F401


COSIM_PROFILE = "ascend950_cannsim_v1"


def require_cosim_profile(profile):
    if profile != COSIM_PROFILE:
        fail(
            f"profile {profile!r} lacks certified peer mapping, acquire_payload and signal visibility; "
            f"only explicit local profile {COSIM_PROFILE!r} is available"
        )
