#include "op/matmul.h"
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/matmul_kernel_i.h"
namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1)
    : LayerFp32Param(device_type, LayerType::kLayerMatmul, "Matmul"),
      dim0_(dim0),
      dim1_(dim1) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
}

base::Status MatmulLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the matmul layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return status;
  }
  return base::error::Success();
}

base::Status MatmulLayer::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0),
                                          nullptr);
  return base::error::Success();
}
}  // namespace op