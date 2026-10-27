#pragma once

#include "kernel_operator.h"
#include "shmem.h"

namespace tl {

// This helper is an atomic scheduler task on AIV0. Its pointers describe real
// accesses; acquire/release-only effects remain in the intrinsic's trailing IR
// operands. A task returns only after all its asynchronous accesses finish.
// InsertSync therefore uses PIPE_S as its completion endpoint, including the
// cross-core ACK edge from an AIC GM-to-L1 DMA. The compiler reserves event 7
// and keeps this scratch disjoint from every local UB allocation.
template <int Kind>
__aicore__ inline void
distributed_task(__gm__ uint8_t *dst, __gm__ uint8_t *src,
                 __gm__ uint32_t *signal, __ubuf__ uint8_t *scratch,
                 uint32_t nbytes, int32_t peer, uint32_t epoch) {
  static_assert(Kind >= 0 && Kind <= 5, "invalid distributed task kind");
  static_assert(ACLSHMEM_CMP_GE == 3, "SHMEM compare ABI changed");
  // The selected transport is MTE only; this is not a rank-wide barrier.
  aclshmemx_mte_quiet();
  if constexpr (Kind == 0 || Kind == 2) {
    aclshmemx_mte_put_nbi<uint8_t>(dst, src, scratch, 4096, nbytes, peer, 7);
    aclshmemx_mte_quiet();
  } else if constexpr (Kind == 1) {
    aclshmemx_mte_get_nbi<uint8_t>(dst, src, scratch, 4096, nbytes, peer, 7);
    aclshmemx_mte_quiet();
  }
  if constexpr (Kind == 2 || Kind == 3) {
    aclshmemx_mte_atomic_set<uint32_t>(signal, epoch, peer);
  } else if constexpr (Kind == 4) {
    aclshmem_uint32_wait_until(signal, ACLSHMEM_CMP_GE, epoch);
  }
  // Publish after payload completion, or acquire after observing the signal.
  // Payload consumers use MTE2, never cached scalar/SIMT GM loads.
  aclshmemx_mte_quiet();
}

} // namespace tl
