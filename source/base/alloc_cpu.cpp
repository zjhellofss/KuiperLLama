#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"
#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define EDGE_HAVE_POSIX_MEMALIGN
#endif
namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
#ifdef EDGE_HAVE_POSIX_MEMALIGN
  void* data = nullptr;
  const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
  int status = posix_memalign(
      (void**)&data, ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)), byte_size);
  if (status != 0) {
    return nullptr;
  }
  return data;
#else
  void* data = malloc(byte_size);
  return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}
void CPUDeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t size) const {
  std::memcpy(dest_ptr, src_ptr, size);
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base