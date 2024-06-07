#include "op/softmax.h"
namespace op {
SoftmaxLayer::SoftmaxLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerSoftmax, "Softmax") {
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
  auto status = check_tensor(get_input(0), device_type_, data_type_);
  if (!status) {
    return status;
  }

  status = check_tensor(get_output(0), device_type_, data_type_);
  if (!status) {
    return status;
  }
  return base::error::Success();
}

}  // namespace op