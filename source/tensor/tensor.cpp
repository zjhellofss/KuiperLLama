#include <numeric>
#include <glog/logging.h>
#include "tensor/tensor.h"

template<typename DataType>
Tensor<DataType>::Tensor(int32_t dim0) {
  dims_.push_back(dim0);
  size_ = dim0;
}

template<typename DataType>
Tensor<DataType>::Tensor(int32_t dim0, int32_t dim1) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
}

template<typename DataType>
Tensor<DataType>::Tensor(int32_t dim0, int32_t dim1, int32_t dim2) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
}

template<typename DataType>
Tensor<DataType>::Tensor(int32_t dim0, int32_t dim1, int32_t dim2,
                         int32_t dim3) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
}

template<typename DataType>
Tensor<DataType>::Tensor(std::vector<int32_t> dims)
    : dims_(std::move(dims)) {
  size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<>());
}

template<typename DataType>
size_t Tensor<DataType>::size() const {
  return this->size_;
}

template<typename DataType>
int32_t Tensor<DataType>::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

template<typename DataType>
bool Tensor<DataType>::assign(std::shared_ptr<Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR)
        << "The buffer parameter in the assign function is null pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (byte_size != buffer->get_byte_size()) {
    return false;
  }
  buffer_ = buffer;
  return true;
}

template<typename DataType>
bool Tensor<DataType>::allocate(std::shared_ptr<DeviceAllocator> allocator, bool need_realloc) {
  if (!allocator) {
    LOG(ERROR) << "The allocator parameter in the allocate function is null "
                  "pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR)
        << "The byte_size parameter in the allocate function is equal to zero!";
    return false;
  }

  if (buffer_ && byte_size == buffer_->get_byte_size()) {
    if (!need_realloc) {
      return true;
    }
  }

  buffer_ = std::make_shared<Buffer>(byte_size, allocator, nullptr);
  if (!buffer_->get_ptr()) {
    LOG(ERROR) << "The memory allocated is a null pointer!";
    return false;
  }
  return true;
}

template<typename DataType>
const std::vector<int32_t> &Tensor<DataType>::dims() const {
  return this->dims_;
}

template<typename DataType>
void Tensor<DataType>::reset_dims(const std::vector<int32_t> &dims) {
  this->dims_ = dims;
  this->size_ = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
}

template<typename DataType>
int32_t Tensor<DataType>::dim_size() const {
  return this->dims_.size();
}

template<typename DataType>
void Tensor<DataType>::reshape(const std::vector<int32_t> &dims) {
  size_t size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
  CHECK_EQ(size, size_);
  this->dims_ = dims;
}

template
class Tensor<float>;