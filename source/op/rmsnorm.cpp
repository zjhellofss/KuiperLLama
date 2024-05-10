#include "op/rmsnorm.h"
#include <armadillo>
namespace op {
RmsNormLayer::RmsNormLayer(int32_t dim)
    : LayerFp32Param(LayerType::kLayerRMSNorm, "RMSNorm"), dim_(dim) {
}

base::Status RmsNormLayer::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  float* in_ptr = get_input(0).ptr<float>();
  float* wei_ptr = get_weight(0).ptr<float>();
  float* out_ptr = get_output(0).ptr<float>();

  arma::fvec in_tensor(in_ptr, dim_, false, true);
  arma::fvec out_tensor(out_ptr, dim_, false, true);
  arma::fvec wei_tensor(wei_ptr, dim_, false, true);

  const float eps = 1e-5f;
  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  const float rsqrt = 1.f / std::sqrt(mean);
  out_tensor = wei_tensor % (rsqrt * in_tensor);
  return base::error::Success();
}

base::Status RmsNormLayer::check() const {
  if (this->input_size() != 1) {
    return base::error::InvalidArgument("The number of input tensors is wrong.");
  }
  if (this->output_size() != 1) {
    return base::error::InvalidArgument("The number of output tensors is wrong.");
  }
  if (this->weight_size() != 1) {
    return base::error::InvalidArgument("The number of output tensors is wrong.");
  }

  const auto& input_tensor = this->get_input(0);
  if (input_tensor.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (input_tensor.device_type() != base::DeviceType::kDeviceCPU) {
    return base::error::InvalidArgument("The input tensor has a wrong device type.");
  }

  if (input_tensor.ptr<int32_t>() == nullptr) {
    return base::error::InvalidArgument("The input tensor is nullptr.");
  }
  if (input_tensor.get_dim(0) != dim_) {
    return base::error::InvalidArgument("The dim0 of input tensor is not equal to the dim.");
  }

  const auto& weight_tensor = this->get_weight(0);
  if (weight_tensor.is_empty()) {
    return base::error::InvalidArgument("The output tensor is empty.");
  }
  if (weight_tensor.device_type() != base::DeviceType::kDeviceCPU) {
    return base::error::InvalidArgument("The weight tensor has a wrong device type.");
  }
  if (weight_tensor.ptr<float>() == nullptr) {
    return base::error::InvalidArgument("The weight tensor is nullptr.");
  }
  if (weight_tensor.get_dim(0) != dim_) {
    return base::error::InvalidArgument(
        "The dim0 of weight tensor is not equal to the vocab size.");
  }

  const auto& output_tensor = this->get_output(0);
  if (output_tensor.is_empty()) {
    return base::error::InvalidArgument("The output tensor is empty.");
  }
  if (output_tensor.device_type() != base::DeviceType::kDeviceCPU) {
    return base::error::InvalidArgument("The output tensor has a wrong device type.");
  }
  if (output_tensor.ptr<float>() == nullptr) {
    return base::error::InvalidArgument("The output tensor is nullptr.");
  }
  if (output_tensor.get_dim(0) != dim_) {
    return base::error::InvalidArgument("The dim0 of output tensor is not equal to the dim.");
  }
  return base::error::Success();
}

}  // namespace op