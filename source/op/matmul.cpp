#include "op/matmul.h"
#include <armadillo>
namespace op {
MatmulLayer::MatmulLayer(int32_t dim0, int32_t dim1)
    : LayerFp32Param(LayerType::kLayerMatmul, "Matmul"), dim0_(dim0), dim1_(dim1) {
}

base::Status MatmulLayer::check() const {
  if (this->input_size() != 1) {
    return base::error::InvalidArgument();
  }
  if (this->output_size() != 1) {
    return base::error::InvalidArgument();
  }
  return base::error::Success();
}

base::Status MatmulLayer::base_forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  auto input = this->get_input(0);
  auto weight = this->get_weight(0);
  auto output = this->get_output(0);

  float* input_ptr = input.ptr<float>();
  float* weight_ptr = weight.ptr<float>();
  float* output_ptr = output.ptr<float>();

  const int32_t in_dim0 = input.get_dim(0);
  const int32_t wei_dim0 = weight.get_dim(0);
  const int32_t wei_dim1 = weight.get_dim(1);
  if (in_dim0 != wei_dim1) {
    return base::error::InternalError("The dim0 of input and the dim1 of weight are not equal.");
  }

  arma::fmat input_vec(input_ptr, 1, in_dim0, false, true);
  arma::fmat weight_mat(weight_ptr, wei_dim1, wei_dim0, false, true);
  arma::fmat output_mat(output_ptr, 1, wei_dim0, false, true);

  output_mat = input_vec * weight_mat;
  // W(dim0, dim1) @ x(dim1) = (dim0) ---> x^t(1,dim1) @ w^t (dim1, dim0)
  return base::error::Success();
}
}  // namespace op