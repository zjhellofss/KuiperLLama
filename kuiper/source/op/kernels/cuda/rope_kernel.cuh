#ifndef ROPE_KERNEL_CU_CUH
#define ROPE_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size,
                    const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                    const tensor::Tensor& input_pos, void* stream);

}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH
