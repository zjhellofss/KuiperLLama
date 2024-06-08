#include "swiglu_kernel.h"
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output) {
  arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false,
                        true);
  arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false,
                        true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false,
                        true);

  input1_vec %= (1.0f / (1.0f + arma::exp(-input1_vec)));
  output_vec = input1_vec % input2_vec;
}

kernel::SwigluKernel kernel::get_swiglu_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return swiglu_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
    return nullptr;
  }
}
