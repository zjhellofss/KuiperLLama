#include <cstdlib>
#include "base/alloc.h"
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
}

void* CPUDeviceAllocator::allocate(size_t size) {
  if (!size) {
    return nullptr;
  }
  void* data = malloc(size);
  return data;
}

void CPUDeviceAllocator::release(void* ptr) {
  if (ptr) {
    free(ptr);
  }
}