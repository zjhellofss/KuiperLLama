#include "op/softmax.h"
namespace op {
SoftmaxLayer::SoftmaxLayer() : Layer(LayerType::kLayerSoftmax, "Softmax") {
}

base::Status SoftmaxLayer::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  tensor::Tensor input = this->get_input(0);
  float* input_ptr = input.ptr<float>();

  int32_t size = static_cast<int32_t>(input.size());
  float max_value = *std::max(input_ptr, input_ptr + size);
  arma::fvec input_mat(input_ptr, size, false, true);
  input_mat = arma::exp(input_mat - max_value);

  float sum_value = arma::sum(input_mat);
  input_mat = input_mat / sum_value;
  return base::error::Success();
}

base::Status SoftmaxLayer::check() const {
  if (this->input_size() != 1) {
    return base::error::InternalError("");
  }
  tensor::Tensor input = this->get_input(0);
  if (input.is_empty()) {
    return base::error::InternalError("");
  }

  if (this->output_size() != 1) {
    return base::error::InternalError("");
  }
  tensor::Tensor output = this->get_output(0);
  if (output.is_empty()) {
    return base::error::InternalError("");
  }
  return base::error::Success();
}

}  // namespace op