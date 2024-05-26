#ifndef LC_INCLUDE_TENSOR_TENSOR_H_
#define LC_INCLUDE_TENSOR_TENSOR_H_
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"
namespace tensor {

class Tensor {
 public:
  explicit Tensor() = default;

  explicit Tensor(base::DataType data_type, int32_t dim0, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                  std::shared_ptr<base::DeviceAllocator> alloc = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                  bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr);

  explicit Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                  bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr);

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

  const std::vector<int32_t>& dims() const;

  std::vector<size_t> strides() const;

  bool assign(std::shared_ptr<base::Buffer> buffer);

  void reset(base::DataType data_type, const std::vector<int32_t>& dims);

  void set_device_type(base::DeviceType device_type);

  base::DeviceType device_type() const;

  bool allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false);

  template <typename T>
  T* ptr(int64_t index);

  template <typename T>
  const T* ptr(int64_t index) const;

  template <typename T>
  T& index(int64_t offset);

  template <typename T>
  void transpose_dim12(Tensor dst);

 private:
  size_t size_ = 0;
  std::vector<int32_t> dims_;
  std::shared_ptr<base::Buffer> buffer_;
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};

template <typename T>
T& Tensor::index(int64_t offset) {
  T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
  return val;
}

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
T* Tensor::ptr(int64_t index) {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
  return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}

template <typename T>
void Tensor::transpose_dim12(Tensor dst) {
  CHECK_EQ(dims_size(), 3);
  CHECK_EQ(is_empty(), false);
  CHECK_EQ(dst.dims_size(), 3);
  CHECK_EQ(dst.is_empty(), false);
  CHECK_EQ(get_dim(0), dst.get_dim(0));
  CHECK_EQ(get_dim(1), dst.get_dim(2));
  CHECK_EQ(get_dim(2), dst.get_dim(1));
  CHECK(device_type() == dst.device_type());
  CHECK(device_type() == base::DeviceType::kDeviceCPU);

  int32_t src_ch = this->get_dim(0);
  int32_t src_row = this->get_dim(1);
  int32_t src_col = this->get_dim(2);
  int32_t dst_row = dst.get_dim(1);
  int32_t dst_col = dst.get_dim(2);
  int32_t plane_size = src_col * src_row;

  T* src_ptr = this->ptr<T>();
  T* dst_ptr = dst.ptr<T>();
  for (int32_t ch = 0; ch < src_ch; ++ch) {
    T* src_ch_ptr = src_ptr + ch * plane_size;
    T* dst_ch_ptr = dst_ptr + ch * plane_size;
    arma::Mat<T> src_mat = arma::Mat<T>(src_ch_ptr, src_col, src_row, false, true);
    arma::Mat<T> dst_mat = arma::Mat<T>(dst_ch_ptr, dst_col, dst_row, false, true);
    dst_mat = src_mat.t();
  }
}
}  // namespace tensor
#endif  // LC_INCLUDE_TENSOR_TENSOR_H_
