#ifndef LC_INCLUDE_BASE_BASE_H_
#define LC_INCLUDE_BASE_BASE_H_
#include <cstdint>
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
};

class Noncopyable {
 protected:
  Noncopyable() = default;

  ~Noncopyable() = default;

 private:
  Noncopyable(const Noncopyable &) = delete;

  Noncopyable &operator=(const Noncopyable &) = delete;
};
#endif  // LC_INCLUDE_BASE_BASE_H_
