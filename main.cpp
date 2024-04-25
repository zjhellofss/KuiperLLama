#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <glog/logging.h>

class Noncopyable {
 protected:
  Noncopyable() {}

  ~Noncopyable() {}
 private:
  Noncopyable(const Noncopyable &);

  Noncopyable &operator=(const Noncopyable &);
};

enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {

  }

  virtual void release(void *ptr) = 0;

  virtual void *allocate(size_t size) = 0;

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {

  }

  void *allocate(size_t size) override {
    if (!size) {
      return nullptr;
    }
    void *data = malloc(size);
    return data;
  }

  void release(void *ptr) override {
    if (ptr) {
      free(ptr);
    }
  }
};

class Buffer : public Noncopyable {
 private:
  size_t byte_size_ = 0;
  void *ptr_ = nullptr;
  bool use_external_ = false;
  std::shared_ptr<DeviceAllocator> allocator_;

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size,
                  std::shared_ptr<DeviceAllocator> allocator,
                  void *ptr = nullptr, bool use_external = false)
      : byte_size_(byte_size), allocator_(allocator), ptr_(ptr), use_external_(use_external) {
    if (!ptr_ && allocator_) {
      use_external_ = false;
      ptr_ = allocator_->allocate(byte_size);
    }
  }

  virtual ~Buffer() {
    if (!use_external_ && ptr_ != nullptr) {
      if (allocator_) {
        allocator_->release(ptr_);
        ptr_ = nullptr;
      }
    }
  }

  void *get_ptr() {
    return ptr_;
  }

  size_t get_byte_size() const {
    return byte_size_;
  }

  const void *get_ptr() const {
    return ptr_;
  }
};

template<typename DataType>
struct Tensor {
  explicit Tensor() = default;

  explicit Tensor(int32_t dim0) {
    dims_.push_back(dim0);
    size_ = dim0;
  }

  explicit Tensor(int32_t dim0, int32_t dim1) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;
  }

  explicit Tensor(int32_t dim0, int32_t dim1, int32_t dim2) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0 * dim1 * dim2;
  }

  explicit Tensor(int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = dim0 * dim1 * dim2 * dim3;
  }

  explicit Tensor(std::vector<int32_t> shapes) : dims_(std::move(shapes)) {
    size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<>());
  }

  template<typename T>
  T *ptr();

  template<typename T>
  const T *ptr() const;

  size_t size() const {
    return this->size_;
  }

  void set_size(size_t size) {
    this->size_ = size;
  }

  int32_t dim_size() const {
    return this->dims_.size();
  }

  int32_t get_dim(int32_t idx) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->dims_.size());
    return this->dims_.at(idx);
  }

  void set_dim(int32_t idx, int32_t dim) {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->dims_.size());
    this->dims_.at(idx) = dim;
  }

  const std::vector<int32_t> &dims() const {
    return this->dims_;
  }

  void reset_dims(const std::vector<int32_t> &dims) {
    this->dims_ = dims;
  }

  bool allocate(size_t size, std::shared_ptr<DeviceAllocator> allocator) {
    if (!size) {
      LOG(ERROR) << "The size parameter in the allocate function is equal to zero!";
      return false;
    }

    if (!allocator) {
      LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
      return false;
    }

    if (size != size_) {
      set_size(size);
    }
    return allocate(allocator);
  }

  bool allocate(std::shared_ptr<DeviceAllocator> allocator) {
    if (!allocator) {
      LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
      return false;
    }

    size_t size = this->size() * sizeof(DataType);
    if (!size) {
      LOG(ERROR) << "The size parameter in the allocate function is equal to zero!";
      return false;
    }

    buffer_ = std::make_shared<Buffer>(size, allocator, nullptr);
    if (!buffer_->get_ptr()) {
      LOG(ERROR) << "The memory allocated is a null pointer!";
      return false;
    }
    return true;
  }

  bool assign(size_t size, std::shared_ptr<Buffer> buffer) {
    if (!buffer) {
      LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
      return false;
    }
    size_ = size;
    buffer_ = buffer;
    return true;
  }

  size_t size_ = 0;
  std::vector<int32_t> dims_;
  std::shared_ptr<Buffer> buffer_;
};

template<typename DataType>
template<typename T>
const T *Tensor<DataType>::ptr() const {
  if (!buffer_) {
    return nullptr;
  }
  return const_cast<const T *>(reinterpret_cast<T *>(buffer_->get_ptr()));
}

template<typename DataType>
template<typename T>
T *Tensor<DataType>::ptr() {
  if (!buffer_) {
    return nullptr;
  }
  return reinterpret_cast<T *>(buffer_->get_ptr());
}

int main() {
  std::shared_ptr<DeviceAllocator> alloc = std::make_shared<CPUDeviceAllocator>();
//  float *a = static_cast<float *>(malloc(sizeof(float) * 3));
//  std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(3, alloc, a, false);
//  Tensor<float> tensor(3);
//  tensor.assign(buffer);
//  free(a);

  int n = 3;
//  std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(n * sizeof(float), alloc, nullptr);
//  Tensor<float> tensor;
//  tensor.assign(3, buffer);

  Tensor<float> t2(12);
  t2.allocate(alloc);
  return 0;
}