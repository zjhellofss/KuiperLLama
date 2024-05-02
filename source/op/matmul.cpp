#include "op/matmul.h"
namespace op {

MatmulLayer::MatmulLayer(int32_t dim0, int32_t dim1)
    : LayerFp32Param(LayerType::kLayerMatmul, "Matmul") {
}

base::Status MatmulLayer::check() {
  if (this->input_size() != 1) {
    return base::error::InvalidArgument();
  }
  if (this->output_size() != 1) {
    return base::error::InvalidArgument();
  }
  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  return base::error::Success();
}
}  // namespace op