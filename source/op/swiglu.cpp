#include "op/swiglu.h"
#include "op/layer.h"
namespace op {
SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, op::LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) {
}

base::Status SwiGLULayer::check() const {
  base::Status status;
  for (int32_t i = 0; i < 2; ++i) {
    status = check_tensor_with_dim(get_input(0), device_type_, data_type_, hidden_dim_);
    if (!status) {
      return status;
    }
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, hidden_dim_);
  if (!status) {
    return status;
  }
  return base::error::Success();
}

base::Status SwiGLULayer::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  arma::fvec input1_vec(input1.ptr<float>(), input1.size(), false, true);
  arma::fvec input2_vec(input2.ptr<float>(), input2.size(), false, true);

  input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));
  input1_vec %= input2_vec;
  return base::error::Success();
}

}  // namespace op
