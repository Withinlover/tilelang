/*!
 * \file tl/op/distributed.cc
 * \brief Source-level distributed protocol operations.
 *
 * BufferRegion annotations remain intact until the distributed verifier and
 * execution profile consume them. Opaque effects prevent ordinary expression
 * simplification from deleting or speculating protocol operations.
 */
#include <tvm/ir/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

#define REGISTER_DISTRIBUTED_OP(Name, Inputs)                                  \
  TVM_REGISTER_OP("tl.distributed." #Name)                                     \
      .set_num_inputs(Inputs)                                                  \
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName",                \
                                          "tl.distributed." #Name)             \
      .set_attr<tirx::TCallEffectKind>("TCallEffectKind",                      \
                                       Integer(tirx::CallEffectKind::kOpaque))

REGISTER_DISTRIBUTED_OP(rank, 1);
REGISTER_DISTRIBUTED_OP(copy_async, 3);
REGISTER_DISTRIBUTED_OP(wait, 2);
REGISTER_DISTRIBUTED_OP(signal, 2);
REGISTER_DISTRIBUTED_OP(wait_signal, 1);
REGISTER_DISTRIBUTED_OP(put_signal, 3);
REGISTER_DISTRIBUTED_OP(fence, 0);
REGISTER_DISTRIBUTED_OP(quiet, 0);

#undef REGISTER_DISTRIBUTED_OP

} // namespace tl
} // namespace tvm
