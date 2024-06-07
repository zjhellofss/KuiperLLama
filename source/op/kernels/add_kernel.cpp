#include "add_kernel.h"
#include <armadillo>
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    tensor::Tensor output) {
  arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false,
                        true);
  arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false,
                        true);
  arma::fvec output_vec(output.ptr<float>(), output.size(), false, true);
  output_vec = input_vec1 + input_vec2;
}

AddKernel get_add_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return add_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

}  // namespace kernel
