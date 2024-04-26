#ifndef LC_INCLUDE_TENSOR_TENSOR_H_
#define LC_INCLUDE_TENSOR_TENSOR_H_
#include <stddef.h>

#include <memory>
#include <vector>

#include "base/base.h"
#include "base/buffer.h"
template <typename DataType>
struct Tensor {
  explicit Tensor() = default;

  explicit Tensor(int32_t dim0);

  explicit Tensor(int32_t dim0, int32_t dim1);

  explicit Tensor(int32_t dim0, int32_t dim1, int32_t dim2);

  explicit Tensor(int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3);

  explicit Tensor(std::vector<int32_t> shapes);

  template <typename T>
  T *ptr();

  template <typename T>
  const T *ptr() const;

  size_t size() const;

  void set_size(size_t size);

  int32_t dim_size() const;

  int32_t get_dim(int32_t idx);

  void set_dim(int32_t idx, int32_t dim);

  const std::vector<int32_t> &dims() const;

  void reset_dims(const std::vector<int32_t> &dims);

  bool allocate(size_t size, std::shared_ptr<DeviceAllocator> allocator);

  bool allocate(std::shared_ptr<DeviceAllocator> allocator);

  bool assign(size_t size, std::shared_ptr<Buffer> buffer);

  size_t size_ = 0;
  std::vector<int32_t> dims_;
  std::shared_ptr<Buffer> buffer_;
};

template <typename DataType>
template <typename T>
const T *Tensor<DataType>::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T *>(reinterpret_cast<T *>(buffer_->get_ptr()));
}

template <typename DataType>
template <typename T>
T *Tensor<DataType>::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T *>(buffer_->get_ptr());
}

#endif  // LC_INCLUDE_TENSOR_TENSOR_H_
