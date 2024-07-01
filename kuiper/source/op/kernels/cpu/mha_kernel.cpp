#include "../cpu/mha_kernel.h"
#include <cuda_runtime_api.h>
#include "../kernels_interface.h"
namespace kernel {
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
    float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);

    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type == base::DeviceType::kDeviceCPU) {
      allocator = base::CPUDeviceAllocatorFactory::get_instance();
    } else {
      allocator = base::CUDADeviceAllocatorFactory::get_instance();
    }

    for (int32_t t = 0; t <= pos; t++) {
      int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
      const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset + cache_offset;
      tensor::Tensor key_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr,
                             const_cast<float*>(key_head_addr));
      tensor::Tensor query_mat(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                               query_head_addr);
      tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, false, nullptr,
                               score_head_addr + t);
      key_mat.set_device_type(device_type);
      query_mat.set_device_type(device_type);
      score_mat.set_device_type(device_type);
      float scale = 1.f / std::sqrt(static_cast<float>(head_size));
      get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
    }

    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos + 1, false, nullptr,
                                     score_head_addr);
    score_head_tensor.set_device_type(device_type);
    get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);

    float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                              config ? config->stream : nullptr, false);
    tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                 output_head_ptr);
    output_tensor.set_device_type(device_type);

    int32_t cache_offset = (h / kv_mul) * head_size;
    float* value_head_addr =
        const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset + cache_offset;
    tensor::Tensor value_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                value_head_addr);
    get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, pos,
                                      head_size, kv_dim, config ? config->stream : nullptr);
  }
}
}  // namespace kernel