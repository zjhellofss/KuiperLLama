#ifndef SWIGLU_KERNEL_I_H
#define SWIGLU_KERNEL_I_H
#include "tensor/tensor.h"
namespace kernel {
typedef void (*SwigluKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output, void* stream);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream = nullptr);
}  // namespace kernel
#endif  // SWIGLU_KERNEL_I_H
