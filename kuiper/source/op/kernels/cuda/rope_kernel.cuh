#ifndef ROPE_KERNEL_CU_CUH
#define ROPE_KERNEL_CU_CUH
#include "tensor/tensor.h"
namespace kernel {

// Original single-token RoPE
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache, void* stream);

// Batched RoPE for multi-token processing (Prefill or multi-sequence Decode)
// input_q: [num_tokens, dim]
// input_k: [num_tokens, kv_dim]
// positions: [num_tokens] - position for each token
void rope_kernel_batch_cu(int32_t num_tokens, int32_t dim, int32_t kv_dim, int32_t head_size,
                          const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                          const tensor::Tensor& positions, const tensor::Tensor& sin_cache,
                          const tensor::Tensor& cos_cache, void* stream);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, cudaStream_t stream);

}  // namespace kernel
#endif  // ROPE_KERNEL_CU_CUH
