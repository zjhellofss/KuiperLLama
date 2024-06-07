#include "op/mha.h"
namespace op {
MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t kv_mul,
                                       int32_t kv_dim, int32_t seq_len, int32_t head_num,
                                       int32_t head_size)
    : Layer(device_type, LayerType::kLayerMHA, "MultiHead"),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size) {
  softmax_ = std::make_unique<op::SoftmaxLayer>(device_type);
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
  tensor::Tensor score_tensor = this->get_input(1);
  tensor::Tensor key_cache_tensor = this->get_input(2);
  tensor::Tensor value_cache_tensor = this->get_input(3);
  tensor::Tensor key_tensor = this->get_input(4);

  int32_t layer_offset = layer_index_ * seq_len_ * kv_dim_;
  for (int32_t h = 0; h < head_num_; ++h) {
    float* score_head_addr = score_tensor.ptr<float>() + h * seq_len_;
    float* query_head_addr = query_tensor.ptr<float>() + h * head_size_;
    arma::fmat query(query_head_addr, 1, head_size_, false, true);
    arma::fmat score(score_head_addr, 1, pos_ + 1, false, true);
    arma::fmat key_mat(key_tensor.ptr<float>(), head_size_, pos_ + 1, false, true);

    for (int32_t t = 0; t <= pos_; t++) {
      float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset + t * kv_dim_ +
                             (h / kv_mul_) * head_size_;
      arma::fvec key_head_vec(key_head_addr, head_size_, false, true);
      key_mat.col(t) = key_head_vec;
    }

    score = (query * key_mat) / std::sqrt(static_cast<float>(head_size_));
    auto score_head_buffer = std::make_shared<base::Buffer>(
        (pos_ + 1) * sizeof(float), nullptr, score_head_addr, true);
    score_head_buffer->set_device_type(device_type_);
    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos_ + 1);
    score_head_tensor.assign(score_head_buffer);
    softmax_->forward_i1o1(score_head_tensor, score_head_tensor);

    arma::fvec output_head(mha_out.ptr<float>() + h * head_size_, head_size_, false,
                           true);
    for (int32_t t = 0; t <= pos_; t++) {
      const float score_value = score.at(t);
      float* value_head_addr = value_cache_tensor.ptr<float>() + layer_offset +
                               t * kv_dim_ + (h / kv_mul_) * head_size_;
      arma::fvec value(value_head_addr, head_size_, false, true);
      if (!t) {
        output_head = score_value * value;
      } else {
        output_head += score_value * value;
      }
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
  base::Status status;
  for (int32_t i = 0; i < 5; ++i) {
    status = check_tensor(get_input(i), device_type_, data_type_);
    if (!status) {
      return status;
    }
  }
  return check_tensor(get_output(0), device_type_, data_type_);
}

}  // namespace op