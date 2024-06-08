#include "op/rope.h"
#include <cmath>
#include "kernels/rope_kernel.h"
namespace op {
RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim,
                     int32_t head_size)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPe"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size) {
}

base::Status RoPELayer::base_forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor input_q = this->get_input(0);
  tensor::Tensor input_k = this->get_input(1);
  tensor::Tensor input_pos = this->get_input(2);
  kernel::get_rope_kernel(device_type_)(dim_, kv_dim_, head_size_, input_q, input_k,
                                        input_pos);
  return base::error::Success();
}

base::Status RoPELayer::check() const {
  auto status = check_tensor_with_dim(get_input(2), device_type_,
                                      base::DataType::kDataTypeInt32, 1);
  if (!status) {
    LOG(ERROR) << "The input tensor 2 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(get_input(1), device_type_, data_type_, kv_dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor 1 error in the add layer.";
    return status;
  }

  status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor 0 error in the add layer.";
    return status;
  }
  return base::error::Success();
}

}  // namespace op