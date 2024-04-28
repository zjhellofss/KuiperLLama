#ifndef LC_INCLUDE_BASE_BASE_H_
#define LC_INCLUDE_BASE_BASE_H_
#include <cstdint>
namespace base {
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 2,
};

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1,
};

inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kDataTypeFp32) {
    return sizeof(float);
  } else if (data_type == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::kDataTypeInt32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

enum class Status : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotValid = 2,
  kParamReadError = 3,
  kWeightReadError = 4,
  kCreateLayerFailed = 5,
};

class Noncopyable {
 protected:
  Noncopyable() = default;

  ~Noncopyable() = default;

 private:
  Noncopyable(const Noncopyable&) = delete;

  Noncopyable& operator=(const Noncopyable&) = delete;
};
}  // namespace base
#endif  // LC_INCLUDE_BASE_BASE_H_
