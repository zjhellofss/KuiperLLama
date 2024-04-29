#ifndef LC_INCLUDE_TENSOR_TENSOR_H_
#define LC_INCLUDE_TENSOR_TENSOR_H_
#include <glog/logging.h>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
namespace tensor {

class Tensor {
 public:
  explicit Tensor() = default;

  explicit Tensor(base::DataType data_type, int32_t dim0);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3);

  explicit Tensor(base::DataType data_type, std::vector<int32_t> dims);

  bool is_empty() const;

  template <typename T>
  T* ptr();

  template <typename T>
  const T* ptr() const;

  void reshape(const std::vector<int32_t>& dims);

  size_t size() const;

  size_t byte_size() const;

  int32_t dims_size() const;

  base::DataType data_type() const;

  int32_t get_dim(int32_t idx) const;

  void reget_dim(const std::vector<int32_t>& dims);

  const std::vector<int32_t>& dims() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  void set_device_type(base::DeviceType device_type);

  base::DeviceType device_type() const;

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

  template <typename T>
  T* index(int64_t index);

  template <typename T>
  const T* index(int64_t index) const;

 private:
  size_t size_ = 0;
  std::vector<int32_t> dims_;
  std::shared_ptr<base::Buffer> buffer_;
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};

template <typename T>
const T* Tensor::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::index(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::index(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

}  // namespace tensor
#endif  // LC_INCLUDE_TENSOR_TENSOR_H_
