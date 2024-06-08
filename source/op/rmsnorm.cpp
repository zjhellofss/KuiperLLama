#include "op/rmsnorm.h"
#include <armadillo>
namespace op {
RmsNormLayer::RmsNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerFp32Param(device_type, LayerType::kLayerRMSNorm, "RMSNorm"), dim_(dim) {
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
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the rmsnorm layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the rmsnorm layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the rmsnorm layer.";
    return status;
  }
  return base::error::Success();
}

}  // namespace op