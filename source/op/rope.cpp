#include "op/rope.h"
#include <cmath>
namespace op {
base::Status RoPELayer::base_forward() {
  base::Status status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor input_q = this->get_input(0);
  tensor::Tensor input_k = this->get_input(1);
  tensor::Tensor input_pos = this->get_input(2);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim_; i += 2) {
    int32_t head_dim = i % head_size_;
    float freq =
        1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size_));
    float val = static_cast<float>(pos) * freq;
    float fcr = std::cos(val);
    float fci = std::sin(val);
    int32_t rotn = i < kv_dim_ ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
    for (int32_t v = 0; v < rotn; v++) {
      float* vec = v == 0 ? input_q.ptr<float>()
                          : input_k.ptr<float>();  // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
  return base::error::Success();
}

base::Status RoPELayer::check() const {
  if (this->input_size() != 3) {
    return base::error::InternalError("The input number is not equal to " + std::to_string(3));
  }
  if (this->output_size() != 1) {
    return base::error::InternalError("The output number is not equal to " + std::to_string(1));
  }
  return base::error::Success();
}

RoPELayer::RoPELayer(int32_t dim, int32_t kv_dim, int32_t head_size)
    : Layer(LayerType::kLayerRoPe, "RoPe"), dim_(dim), kv_dim_(kv_dim), head_size_(head_size) {
}

}  // namespace op