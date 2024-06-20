#ifndef LLAMA_INFER_RMSNORM_KERNEL_H
#define LLAMA_INFER_RMSNORM_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void rmsnorm_kernel_cpu(int32_t dim, const tensor::Tensor& input,
                        const tensor::Tensor& weight, const tensor::Tensor& output,
                        void* stream = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_RMSNORM_KERNEL_H
