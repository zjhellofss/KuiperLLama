#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>

namespace tensor {
template <typename T, typename Tp>
static inline size_t MultiplyAcc(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (need_alloc && alloc) {
    allocate(alloc);
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (need_alloc && alloc) {
    allocate(alloc);
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (need_alloc && alloc) {
    allocate(alloc);
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    allocate(alloc);
  }
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = MultiplyAcc(dims_.begin(), dims_.end(), 1);
}

size_t Tensor::size() const {
  return this->size_;
}

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

base::DeviceType Tensor::device_type() const {
  if (!buffer_) {
    return base::DeviceType::kDeviceUnknown;
  }
  return buffer_->device_type();
}

bool Tensor::assign(size_t byte_size, void* buffer_ptr, bool need_manage) {
  const size_t byte_size_ = this->byte_size();
  if (byte_size != byte_size_) {
    LOG(ERROR) << "The size of buffer is not equal to the tensor!";
    return false;
  }
  buffer_ = std::make_shared<base::Buffer>(byte_size, nullptr, buffer_ptr, !need_manage);
  return true;
}

bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }

  size_t byte_size = this->byte_size();
  if (byte_size != buffer->byte_size()) {
    LOG(ERROR) << "The size of buffer is not equal to the tensor!";
    return false;
  }
  buffer_ = buffer;
  return true;
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
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

  if (buffer_ && byte_size == buffer_->byte_size()) {
    if (!need_realloc) {
      return true;
    }
  }

  buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
  if (!buffer_->ptr()) {
    LOG(ERROR) << "The memory allocated is a null pointer!";
    return false;
  }
  return true;
}

const std::vector<int32_t>& Tensor::dims() const {
  return this->dims_;
}

void Tensor::set_device_type(base::DeviceType device_type) {
  if (buffer_) {
    buffer_->set_device_type(device_type);
  }
}

void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
  this->data_type_ = data_type;
  this->dims_ = dims;
  this->size_ = MultiplyAcc(dims.begin(), dims.end(), 1);
  this->buffer_ = nullptr;
}

int32_t Tensor::dims_size() const {
  return static_cast<int32_t>(dims_.size());
}

base::DataType Tensor::data_type() const {
  return data_type_;
}

void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = MultiplyAcc(dims.begin(), dims.end(), 1);
  if (!buffer_) {
    this->dims_ = dims;
    this->size_ = size;
    return;
  }

  if (size > size_) {
    const size_t byte_size = size * base::DataTypeSize(this->data_type_);
    auto new_buffer = std::make_shared<base::Buffer>(byte_size, buffer_->allocator());
    CHECK(new_buffer->allocate());
    new_buffer->copy_from(buffer_.get());
    this->buffer_ = new_buffer;
  }
  this->dims_ = dims;
  this->size_ = size;
}

size_t Tensor::byte_size() const {
  return this->size() * DataTypeSize(data_type_);
}

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = MultiplyAcc(dims_.begin() + i + 1, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}

bool Tensor::is_empty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}
}  // namespace tensor