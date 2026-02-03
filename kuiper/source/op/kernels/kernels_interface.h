#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H
#include <base/cuda_config.h>
#include "tensor/tensor.h"
namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale, const CudaConfig* config);

typedef void (*MatmulKernelQuant)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t group_size,
                                  const tensor::Tensor& scale, const CudaConfig* config);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size, void* stream);

typedef void (*SwigluKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output, void* stream);

typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                          const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor,
                          const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                          CudaConfig*);

typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

typedef void (*RMSNormKernelDim)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 const tensor::Tensor& output, int32_t dim, void* stream);

typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream);

// Batched RoPE kernel for multi-token processing
typedef void (*RoPEBatchKernel)(int32_t num_tokens, int32_t dim, int32_t kv_dim, int32_t head_size,
                                const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                                const tensor::Tensor& positions, const tensor::Tensor& sin_cache,
                                const tensor::Tensor& cos_cache, void* stream);

typedef void (*ScaleKernel)(float scale, const tensor::Tensor& input, void* stream);

typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input, void* stream);

typedef void (*ScaleSumKernel)(const tensor::Tensor& value, const tensor::Tensor& scale,
                               const tensor::Tensor& output, int t, int size, int stride,
                               void* stream);

// GEMM kernel for multi-token matrix multiplication (Prefill phase)
// Input: [num_tokens, in_dim], Weight: [out_dim, in_dim], Output: [num_tokens, out_dim]
typedef void (*GemmKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, const CudaConfig* config);

// Batched GEMV kernel for multi-sequence decode phase
// Input: [batch_size, in_dim], Weight: [out_dim, in_dim], Output: [batch_size, out_dim]
typedef void (*GemvBatchedKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, const CudaConfig* config);

void softmax_inplace_cpu(const float* input_ptr, size_t size);

AddKernel get_add_kernel(base::DeviceType device_type);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

MHAKernel get_mha_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

RoPEKernel get_rope_kernel(base::DeviceType device_type);

// Get batched RoPE kernel for multi-token processing
RoPEBatchKernel get_rope_batch_kernel(base::DeviceType device_type);

ScaleKernel get_scale_kernel(base::DeviceType device_type);

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream = nullptr);

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type);

// Get GEMM kernel for multi-token matrix multiplication
GemmKernel get_gemm_kernel(base::DeviceType device_type);

// Get batched GEMV kernel for multi-sequence decode
GemvBatchedKernel get_gemv_batched_kernel(base::DeviceType device_type);

}  // namespace kernel
#endif  // KERNELS_INTERFACE_H
