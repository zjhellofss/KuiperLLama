#include "base/alloc.h"
#include <cuda_runtime_api.h>
namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind) const {
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!byte_size) {
    return;
  }
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }
}

}  // namespace base