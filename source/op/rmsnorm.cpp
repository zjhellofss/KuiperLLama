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
  auto inout_status =
      check_inout(1, 1, base::DeviceType::kDeviceCPU, base::DataType::kDataTypeFp32);
  if (!inout_status) {
    return inout_status;
  }
  auto wei_status = check_weight(1, base::DeviceType::kDeviceCPU, base::DataType::kDataTypeFp32);
  if (!wei_status) {
    return wei_status;
  }

  if (this->get_input(0).size() != dim_) {
    return base::error::InternalError("The size of input tensor is not equal to dim.");
  }

  if (this->get_output(0).size() != dim_) {
    return base::error::InternalError("The size of output tensor is not equal to dim.");
  }

  if (this->get_weight(0).size() != dim_) {
    return base::error::InternalError("The size of weight tensor is not equal to dim.");
  }
  return base::error::Success();
}

}  // namespace op