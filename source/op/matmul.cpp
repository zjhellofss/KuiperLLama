#include "op/matmul.h"
#include "kernels/matmul_kernel.h"
namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1)
    : LayerFp32Param(device_type, LayerType::kLayerMatmul, "Matmul"),
      dim0_(dim0),
      dim1_(dim1) {
}

base::Status MatmulLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
  if (!status) {
    return status;
  }

  status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
  if (!status) {
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
  if (!status) {
    return status;
  }
  return base::error::Success();
}

base::Status MatmulLayer::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0));
  return base::error::Success();
}
}  // namespace op