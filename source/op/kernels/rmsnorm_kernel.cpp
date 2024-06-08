#include "rmsnorm_kernel.h"

void rmsnorm_kernel_cpu(int32_t dim, const tensor::Tensor& input,
                        const tensor::Tensor& weight, const tensor::Tensor& output) {
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  const float* out_ptr = output.ptr<float>();

  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

  const float eps = 1e-5f;
  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  const float rsqrt = 1.f / std::sqrt(mean);
  out_tensor = wei_tensor % (rsqrt * in_tensor);
}

kernel::RMSNormKernel kernel::get_rmsnorm_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get an rmsnorm kernel.";
    return nullptr;
  }
}
