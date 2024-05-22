#ifndef LC_INCLUDE_BASE_ALLOC_H_
#define LC_INCLUDE_BASE_ALLOC_H_
#include <memory>
#include "base.h"
namespace base {
class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {
  }

  virtual DeviceType device_type() const {
    return device_type_;
  }

  virtual void release(void* ptr) const = 0;

  virtual void* allocate(size_t size) const = 0;

  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t size) const = 0;

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

  void memcpy(const void* src_ptr, void* dest_ptr, size_t size) const override;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

}  // namespace base
#endif  // LC_INCLUDE_BASE_ALLOC_H_