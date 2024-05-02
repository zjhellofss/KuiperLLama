#include "op/matmul.h"
namespace op {

MatmulLayer::MatmulLayer(int32_t dim0, int32_t dim1)
    : LayerFp32Param(LayerType::kLayerMatmul, "Matmul") {
}

base::Status MatmulLayer::check() {
  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  return base::error::Success();
}
}  // namespace op