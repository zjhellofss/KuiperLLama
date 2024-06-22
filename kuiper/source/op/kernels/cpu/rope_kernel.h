#ifndef LLAMA_INFER_ROPE_KERNEL_H
#define LLAMA_INFER_ROPE_KERNEL_H
#include "tensor/tensor.h"
namespace kernel {
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos, void* stream = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_ROPE_KERNEL_H
