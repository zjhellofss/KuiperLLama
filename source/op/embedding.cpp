#include "op/embedding.h"
#include "kernels/emb_kernel.h"
#include "op/layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerFp32Param(device_type, LayerType::kLayerEmbedding, "Embedding") {
  reset_weight_size(1);
  reset_input_size(2);
  reset_output_size(1);
}

base::Status EmbeddingLayer::check() const {
  const auto& input_tensor = get_input(0);
  const auto& token_size = get_input(1).size();
  if (token_size > input_tensor.size()) {
    return base::error::InvalidArgument(
        "The number of input tensor is greater than seq len.");
  }

  base::Status status = check_tensor_with_dim(input_tensor, device_type_,
                                              base::DataType::kDataTypeInt32, seq_len_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the embedding layer.";
    return status;
  }

  status =
      check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
  if (!status) {
    LOG(ERROR) << "The weight tensor error in the embedding layer.";
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, seq_len_, dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the embedding layer.";
    return status;
  }
  return base::error::Success();
}

base::Status EmbeddingLayer::base_forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }
  kernel::get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0),
                                       vocab_size_);
  return base::StatusCode::kSuccess;
}
}  // namespace op