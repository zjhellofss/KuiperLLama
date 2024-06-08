#ifndef LLAMA_INFER_SWIGLU_KERNEL_H
#define LLAMA_INFER_SWIGLU_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
typedef void (*SwigluKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // LLAMA_INFER_SWIGLU_KERNEL_H
