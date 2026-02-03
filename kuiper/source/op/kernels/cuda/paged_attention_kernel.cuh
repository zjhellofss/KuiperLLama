
#ifndef PAGED_ATTENTION_KERNEL_CU_H
#define PAGED_ATTENTION_KERNEL_CU_H

#include "tensor/tensor.h"
#include <cuda_runtime.h>

namespace kernel {

// 声明 CUDA 实现函数，参数需严格匹配 kernels_interface.h 中的 PagedAttentionKernel
void paged_attention_kernel_cu(
    tensor::Tensor& out,
    const tensor::Tensor& query,
    const tensor::Tensor& key_value_cache,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& context_lens,
    int max_context_len,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale,
    int layer_idx ,
    int num_layers,
    cudaStream_t stream
);

// Batch KV cache write kernel (legacy version)
// Writes K and V for multiple sequences to paged KV cache
// Assumes token_idx == seq_idx (one token per sequence)
// key: [num_seqs, kv_dim]
// value: [num_seqs, kv_dim]
// positions: [num_seqs] - token position for each sequence
void batch_kv_cache_write_cu(
    tensor::Tensor& key_value_cache,  // [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim]
    const tensor::Tensor& key,        // [num_seqs, kv_dim]
    const tensor::Tensor& value,      // [num_seqs, kv_dim]
    const tensor::Tensor& block_tables,  // [num_seqs, max_blocks_per_seq]
    const tensor::Tensor& positions,  // [num_seqs] - token positions
    int num_seqs,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int layer_idx,
    int num_layers,
    cudaStream_t stream
);

// Batch KV cache write kernel v2 (unified version)
// Supports chunked prefill and continuous batching
// Uses seq_ids to map tokens to sequences
// key: [num_tokens, kv_dim]
// value: [num_tokens, kv_dim]
// positions: [num_tokens] - position for each token
// seq_ids: [num_tokens] - sequence index for each token
void batch_kv_cache_write_v2_cu(
    tensor::Tensor& key_value_cache,     // [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim]
    const tensor::Tensor& key,           // [num_tokens, kv_dim]
    const tensor::Tensor& value,         // [num_tokens, kv_dim]
    const tensor::Tensor& block_tables,  // [num_seqs, max_blocks_per_seq]
    const tensor::Tensor& positions,     // [num_tokens] - token positions
    const tensor::Tensor& seq_ids,       // [num_tokens] - sequence index for each token
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int layer_idx,
    int num_layers,
    cudaStream_t stream
);

// Paged attention prefill kernel
// Supports multiple query tokens per sequence with causal mask
// query: [num_tokens, num_heads, head_dim]
// Uses positions for causal masking (each query only attends to positions <= its own)
void paged_attention_prefill_cu(
    tensor::Tensor& out,                    // [num_tokens, num_heads, head_dim]
    const tensor::Tensor& query,            // [num_tokens, num_heads, head_dim]
    const tensor::Tensor& key_value_cache,  // [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim]
    const tensor::Tensor& block_tables,     // [num_seqs, max_blocks_per_seq]
    const tensor::Tensor& context_lens,     // [num_seqs] - context length for each sequence
    const tensor::Tensor& seq_ids,          // [num_tokens] - sequence index for each token
    const tensor::Tensor& positions,        // [num_tokens] - position for each token (for causal mask)
    int num_tokens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale,
    int layer_idx,
    int num_layers,
    cudaStream_t stream
);

} // namespace kernel

#endif // PAGED_ATTENTION_KERNEL_CU_H
