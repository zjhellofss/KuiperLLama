#include "op/add.h"
#include "kernels/add_kernel.h"
namespace op {
VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add") {
}

base::Status VecAddLayer::check() const {
  tensor::Tensor input1 = this->get_input(0);
  tensor::Tensor input2 = this->get_input(0);
  int32_t size = input1.size();
  base::Status status;
  status = check_tensor_with_dim(input1, device_type_, data_type_, size);
  if (!status) {
    return status;
  }

  status = check_tensor_with_dim(input2, device_type_, data_type_, size);
  if (!status) {
    return status;
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
  if (!status) {
    return status;
  }
  return base::error::Success();
}

base::Status VecAddLayer::base_forward() {
  auto status = this->check();
  if (!status) {
    return status;
  }
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (input1.size() != input2.size()) {
    return base::error::InternalError("The input size of two tensors are not match.");
  }
  if (input1.size() != output.size()) {
    return base::error::InternalError("The input and output size are not match.");
  }
  kernel::get_add_kernel(device_type_)(input1, input2, output);
  return base::error::Success();
}

}  // namespace op