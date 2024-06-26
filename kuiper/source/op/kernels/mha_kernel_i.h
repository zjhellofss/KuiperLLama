#ifndef MHA_KERNEL_I_H
#define MHA_KERNEL_I_H
#include "base/cuda_config.h"
#include "tensor/tensor.h"
namespace kernel {
typedef void (*MHAKernel)(
    int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
    int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
    const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
    const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
    const tensor::Tensor& key_tensor, base::DeviceType device_type, CudaConfig*);

MHAKernel get_mha_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // MHA_KERNEL_I_H
