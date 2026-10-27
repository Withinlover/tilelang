"""P0-A: double each local FP32 tile with SIMD, then exchange it on one AIV."""

import tilelang.ascend.language as T
import tilelang.language.distributed as TD

from examples.distributed.example_allgather_gemm import TILE


def vector_exchange(rounds=3, slots=2):
    if not 1 <= rounds <= 10000 or not 1 <= slots <= 10000:
        raise ValueError("rounds and slots must be in 1..10000")
    group = TD.GroupSpec("world", 2)
    channel = TD.ChannelSpec("comm", group)

    @T.prim_func
    def program(
        Input: TD.SymBuffer((rounds, TILE, TILE), "float32", group=group, remote_access="read"),
        Source: TD.SymBuffer((slots, TILE, TILE), "float32", group=group, remote_access="read"),
        Recv: TD.SymBuffer((slots, TILE, TILE), "float32", group=group, remote_access="write"),
        Ready: TD.SignalBuffer((slots,), group=group),
        Ack: TD.SignalBuffer((slots,), group=group),
        Output: TD.SymBuffer((rounds, TILE, TILE), "float32", group=group, remote_access="write"),
        epoch0: T.int32,
    ):
        with T.Kernel(1):
            local = T.alloc_shared((TILE, TILE), "float32")
            doubled = T.alloc_shared((TILE, TILE), "float32")
            rank = TD.rank(group)
            peer = 1 - rank
            for iteration in T.serial(rounds):
                slot = iteration % slots
                epoch = epoch0 + iteration + 1
                if iteration >= slots:
                    TD.wait_signal(Ack[slot], epoch - slots, channel=channel, covers=[])
                T.copy(Input[iteration, :, :], local)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for row in T.serial(TILE):
                        value = T.simd.vld(local[row, 0])
                        result = T.simd.vadd(value, value, mask)
                        T.simd.vsts(doubled[row, 0], result, mask)
                T.copy(doubled, Source[slot, :, :])
                sent = TD.put_signal(
                    Source[slot, :, :],
                    TD.remote(Recv, peer)[slot, :, :],
                    signal=TD.remote(Ready, peer)[slot],
                    value=epoch,
                    channel=channel,
                )
                TD.wait(sent, completion="local")
                TD.wait_signal(Ready[slot], epoch, channel=channel, covers=[Recv[slot, :, :]])
                T.copy(Recv[slot, :, :], local)
                T.copy(local, Output[iteration, :, :])
                TD.signal(TD.remote(Ack, peer)[slot], epoch, channel=channel, release_regions=[Recv[slot, :, :]])
            for iteration in T.serial(max(0, rounds - slots), rounds):
                TD.wait_signal(Ack[iteration % slots], epoch0 + iteration + 1, channel=channel, covers=[])
            TD.quiet(channel=channel)

    return program
