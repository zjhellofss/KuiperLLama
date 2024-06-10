#ifndef LLAMA_INFER_SOFTMAX_KERNEL_H
#define LLAMA_INFER_SOFTMAX_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input);

void softmax_inplace_cpu(const float* input_ptr, size_t size);

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // LLAMA_INFER_SOFTMAX_KERNEL_H
