// Runtime-built adapter: compile against the selected SHMEM and CANN headers.
// No vendor struct layout is duplicated in Python.
#include <acl/acl.h>
#include <dlpack/dlpack.h>
#include <shmem.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {
std::atomic<uint64_t> live_views{0};
struct TensorView {
  DLManagedTensor managed{};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
};
void DeleteView(DLManagedTensor *tensor) {
  auto *view = static_cast<TensorView *>(tensor->manager_ctx);
  delete view;
  --live_views;
}
} // namespace

extern "C" {
int tl_shmem_init(int rank, int size, uint64_t heap_bytes,
                  const char *bootstrap, uint32_t timeout_seconds) {
  if (bootstrap == nullptr ||
      std::strlen(bootstrap) >= ACLSHMEM_MAX_IP_PORT_LEN)
    return ACLSHMEM_INVALID_PARAM;
  if (aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED)
    return ACLSHMEM_INVALID_VALUE;
  aclshmemx_init_attr_t attr;
  attr.my_pe = rank;
  attr.n_pes = size;
  attr.local_mem_size = heap_bytes;
  std::strcpy(attr.ip_port, bootstrap);
  attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
  attr.option_attr.shm_init_timeout = timeout_seconds;
  attr.option_attr.shm_create_timeout = timeout_seconds;
  attr.option_attr.control_operation_timeout = timeout_seconds;
  // Only the loopback bootstrap accepted by Python is used by this profile.
  int status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
  return status == 0 ? aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr)
                     : status;
}
int tl_shmem_finalize() { return aclshmem_finalize(); }
uint64_t tl_shmem_alloc(uint64_t bytes) {
  return reinterpret_cast<uint64_t>(aclshmem_align(64, bytes));
}
void tl_shmem_free(uint64_t pointer) {
  aclshmem_free(reinterpret_cast<void *>(pointer));
}
uint64_t tl_shmem_peer(uint64_t pointer, int rank) {
  return reinterpret_cast<uint64_t>(
      aclshmem_ptr(reinterpret_cast<void *>(pointer), rank));
}
int tl_shmem_zero(uint64_t pointer, uint64_t bytes) {
  return aclrtMemset(reinterpret_cast<void *>(pointer), bytes, 0, bytes);
}
int tl_shmem_h2d(uint64_t pointer, const void *source, uint64_t bytes) {
  return aclrtMemcpy(reinterpret_cast<void *>(pointer), bytes, source, bytes,
                     ACL_MEMCPY_HOST_TO_DEVICE);
}
int tl_shmem_d2h(void *destination, uint64_t pointer, uint64_t bytes) {
  return aclrtMemcpy(destination, bytes, reinterpret_cast<void *>(pointer),
                     bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}
int tl_shmem_stream_create(void **stream) { return aclrtCreateStream(stream); }
int tl_shmem_stream_sync(void *stream, int timeout_ms) {
  return aclrtSynchronizeStreamWithTimeout(stream, timeout_ms);
}
int tl_shmem_stream_destroy(void *stream) { return aclrtDestroyStream(stream); }
int tl_shmem_set_device(int device) { return aclrtSetDevice(device); }
int tl_shmem_physical_device(int device, int *physical_device) {
  return aclrtGetPhyDevIdByUserDevId(device, physical_device);
}
const char *tl_shmem_soc() { return aclrtGetSocName(); }
const char *tl_shmem_error() { return aclGetRecentErrMsg(); }
uint64_t tl_shmem_live_views() { return live_views.load(); }
uint64_t tl_shmem_tensor(uint64_t pointer, int device, uint8_t code,
                         uint8_t bits, int ndim, const int64_t *shape,
                         const int64_t *strides) {
  if (ndim < 1 || pointer == 0)
    return 0;
  auto *view = new (std::nothrow) TensorView;
  if (view == nullptr)
    return 0;
  try {
    view->shape.assign(shape, shape + ndim);
    view->strides.assign(strides, strides + ndim);
  } catch (...) {
    delete view;
    return 0;
  }
  view->managed.dl_tensor = {reinterpret_cast<void *>(pointer),
                             {kDLExtDev, device},
                             ndim,
                             {code, bits, 1},
                             view->shape.data(),
                             view->strides.data(),
                             0};
  view->managed.manager_ctx = view;
  view->managed.deleter = DeleteView;
  ++live_views;
  return reinterpret_cast<uint64_t>(&view->managed);
}
void tl_shmem_delete_tensor(uint64_t pointer) {
  if (pointer != 0)
    DeleteView(reinterpret_cast<DLManagedTensor *>(pointer));
}
}
