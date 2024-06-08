#include "matmul_kernel.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output) {
  const float* input_ptr = input.ptr<float>();
  const float* weight_ptr = weight.ptr<float>();
  const float* output_ptr = output.ptr<float>();

  const int32_t in_dim0 = input.get_dim(0);
  const int32_t wei_dim0 = weight.get_dim(0);
  const int32_t wei_dim1 = weight.get_dim(1);
  CHECK_EQ(in_dim0, wei_dim1);

  arma::fmat input_vec(const_cast<float*>(input_ptr), 1, in_dim0, false, true);
  arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
  arma::fmat output_mat(const_cast<float*>(output_ptr), 1, wei_dim0, false, true);

  output_mat = input_vec * weight_mat;
  // W(dim0, dim1) @ x(dim1) = (dim0) ---> x^t(1,dim1) @ w^t (dim1, dim0)
}

MatMulKernel get_matmul_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return matmul_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get an matmul kernel.";
    return nullptr;
  }
}
}  // namespace kernel