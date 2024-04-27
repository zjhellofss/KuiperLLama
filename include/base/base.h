#ifndef LC_INCLUDE_BASE_BASE_H_
#define LC_INCLUDE_BASE_BASE_H_
#include <cstdint>
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
};

inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kDataTypeFp32) {
    return sizeof(float);
  } else if (data_type == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  } else {
    return 0;
  }
}

class Noncopyable {
 protected:
  Noncopyable() = default;

  ~Noncopyable() = default;

 private:
  Noncopyable(const Noncopyable &) = delete;

  Noncopyable &operator=(const Noncopyable &) = delete;
};
#endif  // LC_INCLUDE_BASE_BASE_H_
