"""Opt-in SDK source translations used by the ordinary Ascend build callback.

Compatibility profiles are deliberately narrow. They never replace a kernel's
computation and they fail on an unfamiliar source shape before invoking Bisheng.
"""


def adapt_source(source, profile, arch):
    if arch != "dav-3510":
        raise ValueError(f"Ascend SDK compatibility {profile!r} requires dav-3510, got {arch!r}")
    if profile == "cann-9.1-gemm64":
        from .distributed.segments import adapt_cann91_gemm64

        return adapt_cann91_gemm64(source)
    if profile == "cann-9.1-shmem":
        from .distributed.transport import adapt_cann91_shmem

        return adapt_cann91_shmem(source)
    raise ValueError(f"Unknown Ascend SDK compatibility profile {profile!r}")
