#ifndef KERNELS_INTERFACE_H
#define KERNELS_INTERFACE_H
#include <base/cuda_config.h>
#include "tensor/tensor.h"
namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale, const CudaConfig* config);

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

typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, void* stream);

typedef void (*ScaleKernel)(float scale, const tensor::Tensor& input, void* stream);

typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor& input, void* stream);

typedef void (*ScaleSumKernel)(const tensor::Tensor& value, const tensor::Tensor& scale,
                               const tensor::Tensor& output, int t, int size, int stride,
                               void* stream);

void softmax_inplace_cpu(const float* input_ptr, size_t size);

AddKernel get_add_kernel(base::DeviceType device_type);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);

MHAKernel get_mha_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

RoPEKernel get_rope_kernel(base::DeviceType device_type);

ScaleKernel get_scale_kernel(base::DeviceType device_type);

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream = nullptr);

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // KERNELS_INTERFACE_H
