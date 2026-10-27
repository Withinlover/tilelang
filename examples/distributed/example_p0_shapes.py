"""Bounded dense AllGather + GEMM, with real device-side M/N/K tile loops.

Every rank owns M x K BF16 A and the same K x N BF16 B. Communication sends
one complete A packet; both ranks compute [A_rank0; A_rank1] @ B. Packing is
an explicit benchmark cost, not a change in the mathematical workload.
"""

import numpy as np

import tilelang.ascend.language as T
import tilelang.language.distributed as TD

TILE = 64


def tile_counts(m, n, k):
    if any(type(value) is not int or value % TILE or not TILE <= value <= 1024 for value in (m, n, k)):
        raise ValueError("M, N and K must each be a multiple of 64 in 64..1024")
    return m // TILE, n // TILE, k // TILE


def allgather_dense_gemm(m=1024, n=1024, k=1024):
    """One READY/ACK packet and two bounded compute regions in one mixed block."""
    mt, nt, kt = tile_counts(m, n, k)
    group = TD.GroupSpec("world", 2)
    channel = TD.ChannelSpec("comm", group)

    @T.prim_func
    def program(
        Send: TD.SymBuffer((mt, kt, TILE, TILE), "bfloat16", group=group, remote_access="read"),
        B: TD.SymBuffer((nt, kt, TILE, TILE), "bfloat16", group=group, remote_access="read"),
        Recv: TD.SymBuffer((mt, kt, TILE, TILE), "bfloat16", group=group, remote_access="write"),
        Ready: TD.SignalBuffer((1,), group=group),
        Ack: TD.SignalBuffer((1,), group=group),
        Output: TD.SymBuffer((2, mt, nt, TILE, TILE), "float32", group=group, remote_access="write"),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            a_l0 = T.alloc_l0a((TILE, TILE), "bfloat16")
            b_l0 = T.alloc_l0b((TILE, TILE), "bfloat16")
            c_l0 = T.alloc_l0c((TILE, TILE), "float32")
            rank = TD.rank(group)
            peer = 1 - rank
            epoch = epoch0 + 1
            sent = TD.put_signal(Send, TD.remote(Recv, peer), signal=TD.remote(Ready, peer)[0], value=epoch, channel=channel)
            TD.wait(sent, completion="local")
            for row in T.serial(mt, annotations={"td.gemm": 1}):
                for col in T.serial(nt):
                    for inner in T.serial(kt):
                        T.copy(Send[row, inner, :, :], a_l1)
                        T.copy(B[col, inner, :, :], b_l1)
                        T.copy(a_l1, a_l0)
                        T.copy(b_l1, b_l0, transpose=True)
                        T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=inner == 0)
                    T.copy(c_l0, Output[rank, row, col, :, :])
            TD.wait_signal(Ready[0], epoch, channel=channel, covers=[Recv])
            for row in T.serial(mt, annotations={"td.gemm": 1}):
                for col in T.serial(nt):
                    for inner in T.serial(kt):
                        T.copy(Recv[row, inner, :, :], a_l1)
                        T.copy(B[col, inner, :, :], b_l1)
                        T.copy(a_l1, a_l0)
                        T.copy(b_l1, b_l0, transpose=True)
                        T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=inner == 0)
                    T.copy(c_l0, Output[peer, row, col, :, :])
            TD.signal(TD.remote(Ack, peer)[0], epoch, channel=channel, release_regions=[Recv])
            TD.wait_signal(Ack[0], epoch, channel=channel, covers=[])
            TD.quiet(channel=channel)

    return program


def bulk_exchange(m=1024, n=1024, k=1024):
    """The host consumes Recv only after complete and before the next prepare."""
    mt, _, kt = tile_counts(m, n, k)
    group = TD.GroupSpec("world", 2)
    channel = TD.ChannelSpec("comm", group)

    @T.prim_func
    def program(
        Send: TD.SymBuffer((mt, kt, TILE, TILE), "bfloat16", group=group, remote_access="read"),
        Recv: TD.SymBuffer((mt, kt, TILE, TILE), "bfloat16", group=group, remote_access="write"),
        Ready: TD.SignalBuffer((1,), group=group),
        Ack: TD.SignalBuffer((1,), group=group),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            rank = TD.rank(group)
            peer = 1 - rank
            epoch = epoch0 + 1
            sent = TD.put_signal(Send, TD.remote(Recv, peer), signal=TD.remote(Ready, peer)[0], value=epoch, channel=channel)
            TD.wait(sent, completion="local")
            TD.wait_signal(Ready[0], epoch, channel=channel, covers=[Recv])
            TD.signal(TD.remote(Ack, peer)[0], epoch, channel=channel, release_regions=[Recv])
            TD.wait_signal(Ack[0], epoch, channel=channel, covers=[])
            TD.quiet(channel=channel)

    return program


def local_dense_gemm(m=1024, n=1024, k=1024):
    """One ordinary grid launch; every output tile retains a K accumulation loop."""
    mt, nt, kt = tile_counts(m, n, k)

    @T.prim_func
    def program(
        A0: T.Tensor((mt, kt, TILE, TILE), "bfloat16"),
        A1: T.Tensor((mt, kt, TILE, TILE), "bfloat16"),
        B: T.Tensor((nt, kt, TILE, TILE), "bfloat16"),
        C: T.Tensor((2, mt, nt, TILE, TILE), "float32"),
    ):
        with T.Kernel(2 * mt * nt) as block:
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            a_l0 = T.alloc_l0a((TILE, TILE), "bfloat16")
            b_l0 = T.alloc_l0b((TILE, TILE), "bfloat16")
            c_l0 = T.alloc_l0c((TILE, TILE), "float32")
            owner = block // (mt * nt)
            row = block // nt % mt
            col = block % nt
            for inner in T.serial(kt):
                if owner == 0:
                    T.copy(A0[row, inner, :, :], a_l1)
                else:
                    T.copy(A1[row, inner, :, :], a_l1)
                T.copy(B[col, inner, :, :], b_l1)
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0, transpose=True)
                T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=inner == 0)
            T.copy(c_l0, C[owner, row, col, :, :])

    return program


def pack_inputs(a, b):
    """Dense BF16 bit patterns -> A[owner,m,k,i,j], B[n,k,i,j]."""
    if a.dtype != np.uint16 or b.dtype != np.uint16 or a.ndim != 3 or a.shape[0] != 2 or b.ndim != 2 or a.shape[2] != b.shape[0]:
        raise ValueError("expected two dense uint16 BF16 A matrices and one compatible B")
    mt, nt, kt = tile_counts(a.shape[1], b.shape[1], a.shape[2])
    packed_a = a.reshape(2, mt, TILE, kt, TILE).transpose(0, 1, 3, 2, 4).copy()
    packed_b = b.reshape(kt, TILE, nt, TILE).transpose(2, 0, 1, 3).copy()
    return packed_a, packed_b


def pack_local_inputs(a, b):
    """Pack just this rank's dense A; the remote rank packs its own input."""
    if a.dtype != np.uint16 or b.dtype != np.uint16 or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected one dense uint16 BF16 A matrix and compatible B")
    mt, nt, kt = tile_counts(a.shape[0], b.shape[1], a.shape[1])
    return (a.reshape(mt, TILE, kt, TILE).transpose(0, 2, 1, 3).copy(), b.reshape(kt, TILE, nt, TILE).transpose(2, 0, 1, 3).copy())


def unpack_output(output):
    if output.ndim != 5 or output.shape[0] != 2 or output.shape[-2:] != (TILE, TILE) or output.dtype != np.float32:
        raise ValueError("expected FP32 blocked [owner,m,n,64,64] output")
    return output.transpose(0, 1, 3, 2, 4).reshape(2 * output.shape[1] * TILE, output.shape[2] * TILE).copy()


def make_dense_inputs(m, n, k, seed):
    tile_counts(m, n, k)
    rng = np.random.default_rng(seed)
    a = (rng.uniform(-0.5, 0.5, (2, m, k)).astype("<f4").view("<u4") >> 16).astype("<u2")
    b = (rng.uniform(-0.5, 0.5, (k, n)).astype("<f4").view("<u4") >> 16).astype("<u2")
    a.flags.writeable = b.flags.writeable = False
    return a, b


def reference_output(a, b):
    """Independent dense FP64 oracle, never based on received/packed buffers."""
    values_a = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    values_b = (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return values_a.reshape(-1, a.shape[-1]) @ values_b


def check_dense_output(actual, expected, *, rtol=2e-5, atol=2e-4):
    if actual.dtype != np.float32 or actual.shape != expected.shape or expected.dtype != np.float64:
        raise ValueError("dense output must be FP32 with the FP64 reference shape")
    finite = np.isfinite(actual)
    error = np.abs(actual.astype(np.float64) - expected)
    matching = finite & (error <= atol + rtol * np.abs(expected))
    return {
        "passed": bool(matching.all()),
        "reference": "dense FP64 matmul of original BF16 inputs",
        "elements": int(actual.size),
        "mismatches": int((~matching).sum()),
        "nonfinite": int((~finite).sum()),
        "max_abs_error": float(error[finite].max()) if finite.any() else None,
        "rtol": rtol,
        "atol": atol,
    }


def workload(m, n, k):
    mt, nt, kt = tile_counts(m, n, k)
    return {
        "m_per_rank": m,
        "n": n,
        "k": k,
        "output_shape_per_rank": [2 * m, n],
        "input_dtype": "bfloat16",
        "output_dtype": "float32",
        "tile": TILE,
        "logical_peer_payload_bytes_per_rank": m * k * 2,
        "flops_per_rank": 4 * m * n * k,
        "output_tiles_per_rank": 2 * mt * nt,
        "k_tiles": kt,
        "payload_batches_per_peer": 1,
    }
