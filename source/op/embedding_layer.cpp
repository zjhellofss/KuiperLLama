#include "op/embedding_layer.h"
#include "op/layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer() : LayerFp32Param(LayerType::kLayerEncode, "Embedding") {
}

base::Status EmbeddingLayer::forward() {
  const auto& input = this->get_input(0);
  const auto& weight = this->get_weight(0);

  return base::Status::kSuccess;
}
}  // namespace op