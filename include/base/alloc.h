#ifndef LC_INCLUDE_BASE_ALLOC_H_
#define LC_INCLUDE_BASE_ALLOC_H_
#include <stddef.h>
#include "base.h"
namespace base {
class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {
  }

  virtual void release(void* ptr) = 0;

  virtual void* allocate(size_t size) = 0;

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void* allocate(size_t size) override;

  void release(void* ptr) override;
};
}  // namespace base
#endif  // LC_INCLUDE_BASE_ALLOC_H_