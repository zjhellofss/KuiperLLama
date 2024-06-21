#ifndef SOFTMAX_KERNEL_I_H
#define SOFTMAX_KERNEL_I_H
#include "tensor/tensor.h"
namespace kernel {
typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input, void* stream);

void softmax_inplace_cpu(const float* input_ptr, size_t size);

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // SOFTMAX_KERNEL_I_H
