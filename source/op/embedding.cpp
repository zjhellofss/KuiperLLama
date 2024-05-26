#include "op/embedding.h"
#include "op/layer.h"
namespace op {
EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                               int32_t vocab_size)
    : dim_(dim),
      seq_len_(seq_len),
      vocab_size_(vocab_size),
      LayerFp32Param(device_type, LayerType::kLayerEmbedding, "Embedding") {
}

base::Status EmbeddingLayer::check() const {
  auto status = check_inout_wei_size(2, 1, 1);
  if (!status) {
    return status;
  }

  status = check_single_input(0, device_type_, base::DataType::kDataTypeInt32);
  if (!status) {
    return status;
  }

  const auto& input_tensor = this->get_input(0);
  const auto& token_size = this->get_input(1).size();
  if (token_size > input_tensor.size()) {
    return base::error::InvalidArgument("The number of input tensor is greater than seq len.");
  }
  if (input_tensor.get_dim(0) != seq_len_) {
    return base::error::InvalidArgument("The dim0 of input tensor is not equal to the seq len.");
  }

  status = check_weight(1, device_type_, base::DataType::kDataTypeFp32);
  if (!status) {
    return status;
  }

  const auto& weight_tensor = this->get_weight(0);
  if (weight_tensor.get_dim(0) != vocab_size_) {
    return base::error::InvalidArgument(
        "The dim0 of weight tensor is not equal to the vocab size.");
  }
  if (weight_tensor.get_dim(1) != dim_) {
    return base::error::InvalidArgument("The dim1 of weight tensor is not equal to the token dim.");
  }

  status = check_single_output(0, device_type_, base::DataType::kDataTypeFp32);
  if (!status) {
    return status;
  }

  const auto& output_tensor = this->get_output(0);
  if (output_tensor.get_dim(0) != seq_len_) {
    return base::error::InvalidArgument("The dim0 of output tensor is not equal to the seq len.");
  }
  if (output_tensor.get_dim(1) != dim_) {
    return base::error::InvalidArgument("The dim1 of output tensor is not equal to the token dim.");
  }
  return base::error::Success();
}

base::Status EmbeddingLayer::base_forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }

  const auto& input_tensor = get_input(0);
  const int32_t input_num = static_cast<int32_t>(get_input(1).size());

  const auto& weight_tensor = get_weight(0);
  const int32_t weight_dim = weight_tensor.get_dim(1);
  const auto& output_tensor = get_output(0);

  CHECK(device_type_ == base::DeviceType::kDeviceCPU);
  const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
  if (!allocator) {
    return base::error::InternalError("Get the memory allocator failed.");
  }
  for (int32_t i = 0; i < input_num; ++i) {
    int32_t token = *input_tensor.ptr<int32_t>(i);
    if (token > vocab_size_) {
      return base::error::InternalError("Token is greater than vocab size.");
    }
    void* dest_ptr = (void*)output_tensor.ptr<float>(i * weight_dim);
    const void* src_ptr = (void*)weight_tensor.ptr<float>(token * weight_dim);
    if (!dest_ptr || !src_ptr) {
      return base::error::InternalError("Invalid src or dest pointer.");
    }
    allocator->memcpy(src_ptr, dest_ptr, weight_dim * sizeof(float));
  }
  return base::StatusCode::kSuccess;
}
}  // namespace op