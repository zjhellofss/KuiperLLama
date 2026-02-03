/**
 * @author    jintang-coder
 *   - PagedAttention: GPU kernels for paged KV cache attention
 *
 * =============================================================================
 * PagedAttention CUDA Kernels
 * =============================================================================
 *
 * 功能: 在分页KV Cache上执行注意力计算，支持Decode/Prefill/Mixed批次
 *
 * KV Cache 内存布局:
 *   kv_cache[Total_Blocks, Num_Layers, 2(K/V), Block_Size, KV_Dim]
 *
 *   访问某个token的Key:
 *     physical_block = block_tables[seq_idx][logical_block]
 *     offset = physical_block * block_stride
 *            + layer_idx * layer_stride
 *            + 0 * kv_plane_stride  // K plane
 *            + token_in_block * kv_dim
 *
 * ---
 * Kernel 1: paged_attention_kernel (Decode专用)
 *
 *   Grid: (num_seqs, num_heads)
 *   Block: 128 threads
 *
 *   每个block处理一个(seq, head)对:
 *     1. 加载query[seq][head]
 *     2. 遍历该seq的所有KV blocks
 *     3. 计算attention scores (online softmax)
 *     4. 输出加权value
 *
 *   例子: seq0有50个tokens，block_size=16
 *     num_logical_blocks = ceil(50/16) = 4
 *     遍历 block_tables[0][0..3] 获取物理块ID
 *     最后一个块只有 50%16=2 个有效token
 *
 * ---
 * Kernel 2: batch_kv_cache_write_kernel (Decode写入)
 *
 *   Grid: (num_seqs)
 *   Block: 256 threads
 *
 *   每个block写入一个序列的新KV:
 *     seq_idx = blockIdx.x
 *     token_pos = positions[seq_idx]
 *     logical_block = token_pos / block_size
 *     physical_block = block_tables[seq_idx][logical_block]
 *     写入 kv_cache[physical_block][layer][K/V][token_in_block]
 *
 * ---
 * Kernel 3: batch_kv_cache_write_kernel_v2 (Prefill/Mixed写入)
 *
 *   Grid: (num_tokens)  // 注意: 按token而非seq
 *   Block: 256 threads
 *
 *   关键区别: 使用 seq_ids[] 映射token到序列
 *
 *   例子: Mixed batch
 *     all_tokens = [t0..t9, t10, t11..t15]  // 16 tokens
 *     seq_ids    = [0,0,0,0,0,0,0,0,0,0, 1, 2,2,2,2,2]
 *     positions  = [0,1,2,3,4,5,6,7,8,9, 25, 0,1,2,3,4]
 *
 *     token_idx=10 (seq1的decode token):
 *       seq_idx = seq_ids[10] = 1
 *       token_pos = positions[10] = 25
 *       logical_block = 25/16 = 1
 *       physical_block = block_tables[1][1]
 *       token_in_block = 25%16 = 9
 *
 * ---
 * Kernel 4: paged_attention_prefill_kernel (Prefill/Mixed注意力)
 *
 *   Grid: (num_tokens, num_heads)  // 每个query token一个block
 *   Block: 128 threads
 *
 *   关键: Causal Mask
 *     每个query只attend到 position <= 自己的位置
 *
 *   例子: Prefill seq0, positions=[0,1,2,3,4]
 *     query at pos=0: attend to [0]
 *     query at pos=1: attend to [0,1]
 *     query at pos=2: attend to [0,1,2]
 *     ...
 *
 *   实现:
 *     effective_context_len = min(context_lens[seq_idx], query_pos + 1)
 *
 * ---
 * Online Softmax (FlashAttention风格):
 *
 *   避免存储完整attention矩阵，逐token更新:
 *
 *   m_prev = -INF, d_prev = 0, acc_out = 0
 *   for each key_token:
 *       score = dot(q, k) * scale
 *       m_new = max(m_prev, score)
 *       exp_score = exp(score - m_new)
 *       correction = exp(m_prev - m_new)
 *       d_prev = d_prev * correction + exp_score
 *       acc_out = acc_out * correction + v * exp_score
 *       m_prev = m_new
 *   output = acc_out / d_prev
 *
 * ---
 * 数据流示意:
 *
 *   BatchMetadata                    GPU Kernels
 *   ┌─────────────────┐              ┌─────────────────────────────┐
 *   │ seq_ids[16]     │─────────────►│ batch_kv_cache_write_v2     │
 *   │ positions[16]   │              │   写入KV到正确的物理块位置   │
 *   └─────────────────┘              └─────────────────────────────┘
 *                                                 │
 *   KVCacheManager                                ▼
 *   ┌─────────────────┐              ┌─────────────────────────────┐
 *   │ block_tables    │─────────────►│ paged_attention_prefill     │
 *   │ context_lens    │              │   从物理块读取KV计算attention│
 *   │ kv_cache        │              └─────────────────────────────┘
 *   └─────────────────┘                           │
 *                                                 ▼
 *                                    ┌─────────────────────────────┐
 *                                    │ Output: [num_tokens, dim]   │
 *                                    └─────────────────────────────┘
 *
 */

#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "paged_attention_kernel.cuh"
#include <base/tick.h>

namespace kernel {
    // kernel 实现 的（v1  , 简化）

__global__ void  paged_attention_kernel_kernel(
    float* __restrict__ out  , // [num_seqs , num_heads , head_dim]
    const float*  __restrict__ query , //[num_seqs][num_heads][head_dim]
    const float*  __restrict__  key_value_cache , // [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim] (KV_Dim = num_kv_heads * head_dim)
    const int* __restrict__ block_tables , 
    const int* __restrict__ context_lens ,
    const int max_context_len,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const float scale, 
    const int layer_idx , 
    const int num_layers

)  {
        int seq_idx  = blockIdx.x; 
        int head_idx = blockIdx.y; 

        int tid = threadIdx.x;   // 对应 head_dim 中的一个元素 
        if(seq_idx >= num_seqs || head_idx >= num_heads) return  ; 

        const long long token_stride = (long long)num_kv_heads * head_dim;
        const long long kv_plane_stride = (long long)block_size * token_stride;
        const long long layer_stride = 2LL * kv_plane_stride;
        const long long block_stride = (long long)num_layers * layer_stride;
        const long long kv_head_idx = head_idx / (num_heads / num_kv_heads);
        const long long layer_offset = layer_stride * layer_idx;

        float q_val = 0.0f;
        if (tid < head_dim) {
          q_val = query[(long long)seq_idx * num_heads * head_dim + (long long)head_idx * head_dim + tid];
        }

        using BlockReduce = cub::BlockReduce<float, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float shared_score;
        float m_prev  = -FLT_MAX;   
        float d_prev =  0.0f ; 
        float acc_out = 0.0f ; 
        int context_len = context_lens[seq_idx]; 
        int num_logical_blocks = (context_len - 1) / block_size + 1;
        for (int i = 0 ; i<num_logical_blocks ; i++) {
            int physical_block_id = block_tables[seq_idx * max_blocks_per_seq +  i ] ; 
            int valid_tokens  = block_size  ; 
            if(i == num_logical_blocks - 1) {
                valid_tokens = context_len %  block_size ;
                if(!valid_tokens) valid_tokens  = block_size; 
            }

            for (int t  =0 ;  t<valid_tokens  ; t++) {
                const long long offset =
                    (long long)physical_block_id * block_stride +
                    layer_offset +
                    (long long)t * token_stride +
                    kv_head_idx * head_dim +
                    tid;

                float dot = 0.0f;
                if (tid < head_dim) {
                  float k_val = key_value_cache[offset];
                  dot = q_val * k_val;
                }

                float dot_sum = BlockReduce(temp_storage).Sum(dot);
                __syncthreads();
                if (tid == 0) {
                  shared_score = dot_sum * scale;
                }
                __syncthreads();
                float score = shared_score;
                float m_new = max(m_prev  , score) ; 
                float exp_score = expf(score  - m_new);  
                float correction = expf(m_prev - m_new ) ; 
                d_prev = d_prev * correction + exp_score ; 
                float v_val = 0.0f;
                if (tid < head_dim) {
                  v_val = key_value_cache[offset + kv_plane_stride];
                }
                acc_out = acc_out * correction + v_val * exp_score ; 
                m_prev = m_new ; 
            }
            
        }     
        if (tid < head_dim) {
          float final_out = acc_out / d_prev;
          out[(long long)seq_idx * num_heads * head_dim + (long long)head_idx * head_dim + tid] = final_out;
        }
}

void paged_attention_kernel_cu(tensor::Tensor& out, const tensor::Tensor& query,
                               const tensor::Tensor& key_value_cache,
                               const tensor::Tensor& block_tables,
                               const tensor::Tensor& context_lens, int max_context_len,
                               int num_seqs, int num_heads, int num_kv_heads, int head_dim,
                               int block_size, int max_blocks_per_seq, float scale, int layer_idx
                               ,int num_layers,
                               cudaStream_t stream) {
  dim3 grid(num_seqs, num_heads);
  dim3 block(128);
  CHECK_EQ(out.is_empty(), false);
  CHECK_EQ(query.is_empty(), false);
  CHECK_EQ(key_value_cache.is_empty(), false);
  CHECK_EQ(block_tables.is_empty(), false);
  CHECK_EQ(context_lens.is_empty(), false);
  paged_attention_kernel_kernel<<<grid, block, 0, stream>>>(
      out.ptr<float>(), query.ptr<float>(), key_value_cache.ptr<float>(),
      block_tables.ptr<int>(), context_lens.ptr<int>(), max_context_len, num_seqs, num_heads,
      num_kv_heads, head_dim, block_size, max_blocks_per_seq, scale, layer_idx, num_layers);
}

// Batch KV cache write kernel
// Each thread block handles one sequence, threads handle kv_dim elements
__global__ void batch_kv_cache_write_kernel(
    float* __restrict__ key_value_cache,  // [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim]
    const float* __restrict__ key,        // [num_seqs, kv_dim]
    const float* __restrict__ value,      // [num_seqs, kv_dim]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ positions,    // [num_seqs]
    const int num_seqs,
    const int kv_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const int layer_idx,
    const int num_layers
) {
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (seq_idx >= num_seqs) return;

    // Get token position for this sequence
    int token_pos = positions[seq_idx];

    // Calculate which block and position within block
    int logical_block = token_pos / block_size;
    int token_in_block = token_pos % block_size;

    // Get physical block ID from block table
    int physical_block_id = block_tables[seq_idx * max_blocks_per_seq + logical_block];

    // Calculate strides for KV cache layout: [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim]
    const long long token_stride = (long long)kv_dim;
    const long long kv_plane_stride = (long long)block_size * token_stride;
    const long long layer_stride = 2LL * kv_plane_stride;
    const long long block_stride = (long long)num_layers * layer_stride;

    // Base offset for this block, layer, and token position
    const long long base_offset = (long long)physical_block_id * block_stride +
                                  (long long)layer_idx * layer_stride +
                                  (long long)token_in_block * token_stride;

    // Key offset (kv_plane index 0)
    const long long key_offset = base_offset;
    // Value offset (kv_plane index 1)
    const long long value_offset = base_offset + kv_plane_stride;

    // Input offset for this sequence
    const long long input_offset = (long long)seq_idx * kv_dim;

    // Copy key and value elements (each thread handles multiple elements if kv_dim > blockDim.x)
    for (int i = tid; i < kv_dim; i += blockDim.x) {
        key_value_cache[key_offset + i] = key[input_offset + i];
        key_value_cache[value_offset + i] = value[input_offset + i];
    }
}

void batch_kv_cache_write_cu(
    tensor::Tensor& key_value_cache,
    const tensor::Tensor& key,
    const tensor::Tensor& value,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& positions,
    int num_seqs,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int layer_idx,
    int num_layers,
    cudaStream_t stream
) {
    CHECK_EQ(key_value_cache.is_empty(), false);
    CHECK_EQ(key.is_empty(), false);
    CHECK_EQ(value.is_empty(), false);
    CHECK_EQ(block_tables.is_empty(), false);
    CHECK_EQ(positions.is_empty(), false);

    int kv_dim = num_kv_heads * head_dim;

    // Launch one block per sequence, 256 threads per block
    dim3 grid(num_seqs);
    dim3 block(256);

    batch_kv_cache_write_kernel<<<grid, block, 0, stream>>>(
        key_value_cache.ptr<float>(),
        key.ptr<float>(),
        value.ptr<float>(),
        block_tables.ptr<int>(),
        positions.ptr<int>(),
        num_seqs,
        kv_dim,
        block_size,
        max_blocks_per_seq,
        layer_idx,
        num_layers
    );
}

// ============================================================================
// Batch KV cache write kernel v2 (unified version)
// Supports chunked prefill and continuous batching
// Uses seq_ids to map tokens to sequences
// ============================================================================
__global__ void batch_kv_cache_write_kernel_v2(
    float* __restrict__ key_value_cache,  // [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim]
    const float* __restrict__ key,        // [num_tokens, kv_dim]
    const float* __restrict__ value,      // [num_tokens, kv_dim]
    const int* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ positions,    // [num_tokens]
    const int* __restrict__ seq_ids,      // [num_tokens] - sequence index for each token
    const int num_tokens,
    const int kv_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const int layer_idx,
    const int num_layers
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    // Get sequence index from seq_ids (key difference from v1)
    int seq_idx = seq_ids[token_idx];
    int token_pos = positions[token_idx];

    // Calculate which block and position within block
    int logical_block = token_pos / block_size;
    int token_in_block = token_pos % block_size;

    // Get physical block ID from block table
    int physical_block_id = block_tables[seq_idx * max_blocks_per_seq + logical_block];

    // Calculate strides for KV cache layout: [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim]
    const long long token_stride = (long long)kv_dim;
    const long long kv_plane_stride = (long long)block_size * token_stride;
    const long long layer_stride = 2LL * kv_plane_stride;
    const long long block_stride = (long long)num_layers * layer_stride;

    // Base offset for this block, layer, and token position
    const long long base_offset = (long long)physical_block_id * block_stride +
                                  (long long)layer_idx * layer_stride +
                                  (long long)token_in_block * token_stride;

    // Key offset (kv_plane index 0)
    const long long key_offset = base_offset;
    // Value offset (kv_plane index 1)
    const long long value_offset = base_offset + kv_plane_stride;

    // Input offset for this token
    const long long input_offset = (long long)token_idx * kv_dim;

    // Copy key and value elements
    for (int i = tid; i < kv_dim; i += blockDim.x) {
        key_value_cache[key_offset + i] = key[input_offset + i];
        key_value_cache[value_offset + i] = value[input_offset + i];
    }
}

void batch_kv_cache_write_v2_cu(
    tensor::Tensor& key_value_cache,
    const tensor::Tensor& key,
    const tensor::Tensor& value,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& positions,
    const tensor::Tensor& seq_ids,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int layer_idx,
    int num_layers,
    cudaStream_t stream
) {
    CHECK_EQ(key_value_cache.is_empty(), false);
    CHECK_EQ(key.is_empty(), false);
    CHECK_EQ(value.is_empty(), false);
    CHECK_EQ(block_tables.is_empty(), false);
    CHECK_EQ(positions.is_empty(), false);
    CHECK_EQ(seq_ids.is_empty(), false);

    int kv_dim = num_kv_heads * head_dim;

    // Launch one block per token, 256 threads per block
    dim3 grid(num_tokens);
    dim3 block(256);

    batch_kv_cache_write_kernel_v2<<<grid, block, 0, stream>>>(
        key_value_cache.ptr<float>(),
        key.ptr<float>(),
        value.ptr<float>(),
        block_tables.ptr<int>(),
        positions.ptr<int>(),
        seq_ids.ptr<int>(),
        num_tokens,
        kv_dim,
        block_size,
        max_blocks_per_seq,
        layer_idx,
        num_layers
    );
}

// ============================================================================
// Paged attention prefill kernel
// Supports multiple query tokens per sequence with causal mask
// Each query only attends to positions <= its own position
// ============================================================================
__global__ void paged_attention_prefill_kernel_impl(
    float* __restrict__ out,               // [num_tokens, num_heads, head_dim]
    const float* __restrict__ query,       // [num_tokens, num_heads, head_dim]
    const float* __restrict__ key_value_cache,  // [Total_Blocks, Num_Layers, 2, Block_Size, KV_Dim]
    const int* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int* __restrict__ seq_ids,       // [num_tokens]
    const int* __restrict__ positions,     // [num_tokens] - for causal mask
    const int num_tokens,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks_per_seq,
    const float scale,
    const int layer_idx,
    const int num_layers
) {
    // Grid: (num_tokens, num_heads)
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    int seq_idx = seq_ids[token_idx];
    int query_pos = positions[token_idx];  // Current query's position

    // Causal mask: only attend to positions <= query_pos
    // Also limited by context_lens[seq_idx]
    int context_len = context_lens[seq_idx];
    int effective_context_len = min(context_len, query_pos + 1);

    if (effective_context_len <= 0) {
        // No keys to attend to, output zeros
        if (tid < head_dim) {
            out[(long long)token_idx * num_heads * head_dim + (long long)head_idx * head_dim + tid] = 0.0f;
        }
        return;
    }

    // Calculate strides
    const long long token_stride = (long long)num_kv_heads * head_dim;
    const long long kv_plane_stride = (long long)block_size * token_stride;
    const long long layer_stride = 2LL * kv_plane_stride;
    const long long block_stride = (long long)num_layers * layer_stride;
    const long long kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const long long layer_offset = layer_stride * layer_idx;

    // Load query value
    float q_val = 0.0f;
    if (tid < head_dim) {
        q_val = query[(long long)token_idx * num_heads * head_dim + (long long)head_idx * head_dim + tid];
    }

    // FlashAttention-style online softmax
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_score;

    float m_prev = -FLT_MAX;
    float d_prev = 0.0f;
    float acc_out = 0.0f;

    int num_logical_blocks = (effective_context_len - 1) / block_size + 1;

    for (int i = 0; i < num_logical_blocks; i++) {
        int physical_block_id = block_tables[seq_idx * max_blocks_per_seq + i];
        int valid_tokens = block_size;
        if (i == num_logical_blocks - 1) {
            valid_tokens = effective_context_len % block_size;
            if (!valid_tokens) valid_tokens = block_size;
        }

        for (int t = 0; t < valid_tokens; t++) {
            const long long offset =
                (long long)physical_block_id * block_stride +
                layer_offset +
                (long long)t * token_stride +
                kv_head_idx * head_dim +
                tid;

            float dot = 0.0f;
            if (tid < head_dim) {
                float k_val = key_value_cache[offset];
                dot = q_val * k_val;
            }

            float dot_sum = BlockReduce(temp_storage).Sum(dot);
            __syncthreads();
            if (tid == 0) {
                shared_score = dot_sum * scale;
            }
            __syncthreads();

            float score = shared_score;
            float m_new = max(m_prev, score);
            float exp_score = expf(score - m_new);
            float correction = expf(m_prev - m_new);
            d_prev = d_prev * correction + exp_score;

            float v_val = 0.0f;
            if (tid < head_dim) {
                v_val = key_value_cache[offset + kv_plane_stride];
            }
            acc_out = acc_out * correction + v_val * exp_score;
            m_prev = m_new;
        }
    }

    if (tid < head_dim) {
        float final_out = (d_prev > 0.0f) ? (acc_out / d_prev) : 0.0f;
        out[(long long)token_idx * num_heads * head_dim + (long long)head_idx * head_dim + tid] = final_out;
    }
}

void paged_attention_prefill_cu(
    tensor::Tensor& out,
    const tensor::Tensor& query,
    const tensor::Tensor& key_value_cache,
    const tensor::Tensor& block_tables,
    const tensor::Tensor& context_lens,
    const tensor::Tensor& seq_ids,
    const tensor::Tensor& positions,
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
) {
    CHECK_EQ(out.is_empty(), false);
    CHECK_EQ(query.is_empty(), false);
    CHECK_EQ(key_value_cache.is_empty(), false);
    CHECK_EQ(block_tables.is_empty(), false);
    CHECK_EQ(context_lens.is_empty(), false);
    CHECK_EQ(seq_ids.is_empty(), false);
    CHECK_EQ(positions.is_empty(), false);

    dim3 grid(num_tokens, num_heads);
    dim3 block(128);

    paged_attention_prefill_kernel_impl<<<grid, block, 0, stream>>>(
        out.ptr<float>(),
        query.ptr<float>(),
        key_value_cache.ptr<float>(),
        block_tables.ptr<int>(),
        context_lens.ptr<int>(),
        seq_ids.ptr<int>(),
        positions.ptr<int>(),
        num_tokens,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        scale,
        layer_idx,
        num_layers
    );
}

}