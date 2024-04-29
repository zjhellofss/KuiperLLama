#include "op/embedding_layer.h"
#include "op/layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer() : LayerFp32Param(LayerType::kLayerEncode, "Embedding") {
}

base::Status EmbeddingLayer::check() {
  if (this->input_size() != 2) {
    return base::Status::kInferErrorInput;
  }
  if (this->output_size() != 1) {
    return base::Status::kInferErrorOutput;
  }
  if (this->weight_size() != 1) {
    return base::Status::kInferErrorWeight;
  }

  const auto& input = this->get_input(0);
  if (input.is_empty()) {
    return base::Status::kInferErrorInput;
  }
  if (input.device_type() != base::DeviceType::kDeviceCPU) {
    return base::Status::kInferErrorInput;
  }

  const auto& input_num = this->get_input(1).size();
  if (input_num > input.size()) {
    return base::Status::kInferErrorInput;
  }

  if (input.ptr<int32_t>() == nullptr) {
    return base::Status::kInferErrorInput;
  }

  const auto& weight_tensor = this->get_weight(0);
  if (weight_tensor.is_empty()) {
    return base::Status::kInferErrorWeight;
  }
  if (weight_tensor.device_type() != base::DeviceType::kDeviceCPU) {
    return base::Status::kInferErrorWeight;
  }
  if (weight_tensor.ptr<float>() == nullptr) {
    return base::Status::kInferErrorWeight;
  }

  const auto& output_tensor = this->get_output(0);
  if (output_tensor.is_empty()) {
    return base::Status::kInferErrorOutput;
  }
  if (output_tensor.device_type() != base::DeviceType::kDeviceCPU) {
    return base::Status::kInferErrorWeight;
  }
  if (output_tensor.ptr<float>() == nullptr) {
    return base::Status::kInferErrorOutput;
  }
  return base::Status::kSuccess;
}

base::Status EmbeddingLayer::forward() {
  const auto& status = check();
  if (status != base::Status::kSuccess) {
    return status;
  }

  const auto& input_tensor = get_input(0);
  const int32_t input_num = static_cast<int32_t>(get_input(1).size());

  const auto& weight_tensor = get_weight(0);
  const int32_t weight_dim = weight_tensor.get_dim(1);
  const auto& output_tensor = get_output(0);

  const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
  for (int32_t i = 0; i < input_num; ++i) {
    int32_t token = *input_tensor.index<int32_t>(i);
    allocator->memcpy((void*)weight_tensor.index<float>(token * weight_dim),
                      (void*)output_tensor.index<float>(i * weight_dim),
                      weight_dim * sizeof(float));
  }
  const float* output_ptr = output_tensor.ptr<float>();
  return base::Status::kSuccess;
}
}  // namespace op