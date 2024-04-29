#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"
namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUDeviceAllocator::allocate(size_t size) const {
  if (!size) {
    return nullptr;
  }
  void* data = malloc(size);
  return data;
}

void CPUDeviceAllocator::release(void* ptr) const {
  if (ptr) {
    free(ptr);
  }
}

void CPUDeviceAllocator::memcpy(void* src_ptr, void* dest_ptr, size_t size) const {
  std::memcpy(dest_ptr, src_ptr, size);
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base