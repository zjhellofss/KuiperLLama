#include "op/embedding_layer.h"
#include "op/layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer() : LayerFp32Param(LayerType::kLayerEncode, "Embedding") {
}

base::Status EmbeddingLayer::forward() {
  if (this->input_size() != 2) {
    return base::Status::kErrorInputSize;
  }
  if (this->output_size() != 1) {
    return base::Status::kErrorOutputSize;
  }
  if (this->weight_size() != 1) {
    return base::Status::kErrorWeightSize;
  }

  const auto& input = this->get_input(0);
  const auto& input_num_tensor = this->get_input(1);

  const float* input_ptr = input.ptr<float>();
  if (input_ptr == nullptr) {
    return base::Status::kErrorNullPointer;
  }

  const auto& input_num = input_num_tensor.size();
  const auto& weight_tensor = this->get_weight(0);
  const float* weight_ptr = weight_tensor.ptr<float>();
  if (weight_ptr == nullptr) {
    return base::Status::kErrorNullPointer;
  }

  return base::Status::kSuccess;
}
}  // namespace op