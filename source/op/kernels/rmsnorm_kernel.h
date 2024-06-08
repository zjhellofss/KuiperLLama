#ifndef LLAMA_INFER_RMSNORM_KERNEL_H
#define LLAMA_INFER_RMSNORM_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
typedef void (*RMSNormKernel)(int32_t dim, const tensor::Tensor& input,
                              const tensor::Tensor& weight, const tensor::Tensor& output);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // LLAMA_INFER_RMSNORM_KERNEL_H
