#ifndef LLAMA_INFER_MATMUL_KERNEL_H
#define LLAMA_INFER_MATMUL_KERNEL_H
#include "base/blas_helper.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale = 1.f,
                       const BlasCudaConfig* config = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_MATMUL_KERNEL_H
