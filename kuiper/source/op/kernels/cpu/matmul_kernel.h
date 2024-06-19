#ifndef LLAMA_INFER_MATMUL_KERNEL_H
#define LLAMA_INFER_MATMUL_KERNEL_H
#include "base/base.h"
#include "tensor/tensor.h"
namespace kernel {
typedef void (*MatMulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output);

MatMulKernel get_matmul_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // LLAMA_INFER_MATMUL_KERNEL_H
