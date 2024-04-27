#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>

template <typename T, typename Tp>
static inline size_t MultiplyAccumulate(T begin, T end, Tp init) {
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

Tensor::Tensor(DataType data_type, int32_t dim0) : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1) : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
}

Tensor::Tensor(DataType data_type, std::vector<int32_t> dims)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = MultiplyAccumulate(dims.begin(), dims.end(), 1);
}

size_t Tensor::size() const { return this->size_; }

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

bool Tensor::assign(std::shared_ptr<Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (byte_size != buffer->get_byte_size()) {
    return false;
  }
  buffer_ = buffer;
  return true;
}

bool Tensor::allocate(std::shared_ptr<DeviceAllocator> allocator, bool need_realloc) {
  if (!allocator) {
    LOG(ERROR) << "The allocator parameter in the allocate function is null "
                  "pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (!byte_size) {
    LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
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

const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

void Tensor::reset_dims(const std::vector<int32_t>& dims) {
  this->dims_ = dims;
  this->size_ = MultiplyAccumulate(dims.begin(), dims.end(), 1);
}

int32_t Tensor::dim_size() const { return static_cast<int32_t>(dims_.size()); }

DataType Tensor::data_type() const { return data_type_; }

void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = MultiplyAccumulate(dims.begin(), dims.end(), 1);
  CHECK_EQ(size, size_);
  this->dims_ = dims;
}

size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = MultiplyAccumulate(dims_.begin() + i + 1, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}
