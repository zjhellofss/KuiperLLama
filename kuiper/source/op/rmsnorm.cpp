#include "op/rmsnorm.h"
#include <armadillo>
#include "kernels/cpu/rmsnorm_kernel.h"
#include "kernels/rms_kernel_i.h"
namespace op {
RmsNormLayer::RmsNormLayer(base::DeviceType device_type, int32_t dim)
    : LayerFp32Param(device_type, LayerType::kLayerRMSNorm, "RMSNorm"), dim_(dim) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status RmsNormLayer::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);
  kernel::get_rmsnorm_kernel(device_type_)(input, weight, output, nullptr);
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