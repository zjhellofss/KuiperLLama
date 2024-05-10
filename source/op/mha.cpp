
#include "op/mha.h"
namespace op {
MultiHeadAttention::MultiHeadAttention(int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                       int32_t head_num, int32_t head_size)
    : Layer(LayerType::kLayerMHA, "MultiHead"),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size) {
  softmax_ = std::make_unique<op::SoftmaxLayer>();
  softmax_->reset_input_size(1);
  softmax_->reset_output_size(1);
}

base::Status MultiHeadAttention::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  tensor::Tensor mha_out = this->get_output(0);
  tensor::Tensor query_tensor = this->get_input(0);
  tensor::Tensor attn_tensor = this->get_input(1);
  tensor::Tensor key_cache_tensor = this->get_input(2);
  tensor::Tensor value_cache_tensor = this->get_input(3);

  int32_t layer_offset = layer_index_ * seq_len_ * kv_dim_;
  for (int32_t h = 0; h < head_num_; ++h) {
    float* attn_head_addr = attn_tensor.ptr<float>() + h * seq_len_;
    float* query_head_addr = query_tensor.ptr<float>() + h * head_size_;
    arma::fvec query(query_head_addr, head_size_, false, true);

    for (int32_t t = 0; t <= pos_; t++) {
      float* key_head_addr =
          key_cache_tensor.ptr<float>() + layer_offset + t * kv_dim_ + (h / kv_mul_) * head_size_;
      arma::frowvec key(key_head_addr, head_size_, false, true);
      const float score = arma::as_scalar(key * query) / std::sqrt(static_cast<float>(head_size_));
      attn_head_addr[t] = score;
    }

    auto attn_head_buffer =
        std::make_shared<base::Buffer>((pos_ + 1) * sizeof(float), nullptr, attn_head_addr, true);
    attn_head_buffer->set_device_type(base::DeviceType::kDeviceCPU);
    tensor::Tensor attn_head_tensor(base::DataType::kDataTypeFp32, pos_ + 1);
    attn_head_tensor.assign(attn_head_buffer);
    softmax_->forward_i1o1(attn_head_tensor, attn_head_tensor);

    arma::fvec output_head(mha_out.ptr<float>() + h * head_size_, head_size_, false, true);
    output_head.zeros();
    for (int32_t t = 0; t <= pos_; t++) {
      const float attn_score = attn_head_addr[t];
      float* value_head_addr =
          value_cache_tensor.ptr<float>() + layer_offset + t * kv_dim_ + (h / kv_mul_) * head_size_;
      arma::fvec value(value_head_addr, head_size_, false, true);
      output_head += attn_score * value;
    }
  }
  return base::error::Success();
}

void MultiHeadAttention::set_pos(int32_t pos) {
  this->pos_ = pos;
}

void MultiHeadAttention::set_layer_index(int32_t layer_index) {
  this->layer_index_ = layer_index;
}

base::Status MultiHeadAttention::check() const {
  if (this->input_size() != 4) {
    return base::error::InternalError("");
  }
  if (this->output_size() != 1) {
    return base::error::InternalError("");
  }
  return base::error::Success();
}

}  // namespace op