#ifndef ROPE_KERNEL_I_H
#define ROPE_KERNEL_I_H
#include "tensor/tensor.h"
namespace kernel {
typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, void* stream);

RoPEKernel get_rope_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // ROPE_KERNEL_I_H
