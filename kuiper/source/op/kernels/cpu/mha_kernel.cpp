#include "../cpu/mha_kernel.h"
#include <cuda_runtime_api.h>
#include "../add_kernel_i.h"
#include "../matmul_kernel_i.h"
#include "../scale_tensor_i.h"
#include "../softmax_kernel_i.h"
namespace kernel {

void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor,
                const tensor::Tensor& value_cache_tensor,
                const tensor::Tensor& key_tensor, base::DeviceType device_type,
                void* stream) {
  UNUSED(stream);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  for (int32_t h = 0; h < head_num; ++h) {
    float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
    float* query_head_addr =
        const_cast<float*>(query_tensor.ptr<float>() + h * head_size);

    const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    for (int32_t t = 0; t <= pos; t++) {
      const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset +
                                   t * kv_dim + (h / kv_mul) * head_size;

      if (device_type == base::DeviceType::kDeviceCPU) {
        allocator->memcpy(key_head_addr,
                          const_cast<float*>(key_tensor.ptr<float>(t * head_size)),
                          head_size * sizeof(float), base::MemcpyKind::kMemcpyCPU2CPU);
      } else {
        allocator->memcpy(
            key_head_addr, const_cast<float*>(key_tensor.ptr<float>(t * head_size)),
            head_size * sizeof(float), base::MemcpyKind::kMemcpyCUDA2CUDA, stream);
      }
    }

    tensor::Tensor key_mat(base::DataType::kDataTypeFp32, head_size, pos + 1, false,
                           nullptr, const_cast<float*>(key_tensor.ptr<float>()));
    tensor::Tensor query_mat(base::DataType::kDataTypeFp32, 1, head_size, false, nullptr,
                             query_head_addr);
    tensor::Tensor score_mat(base::DataType::kDataTypeFp32, 1, pos + 1, false, nullptr,
                             score_head_addr);
    key_mat.set_device_type(device_type);
    query_mat.set_device_type(device_type);
    score_mat.set_device_type(device_type);

    float scale = 1.f / std::sqrt(static_cast<float>(head_size));
    get_matmul_kernel(device_type)(key_mat, query_mat, score_mat, scale, nullptr);

    auto score_head_buffer = std::make_shared<base::Buffer>(
        (pos + 1) * sizeof(float), nullptr, const_cast<float*>(score_head_addr), true);
    score_head_buffer->set_device_type(device_type);
    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos + 1);
    score_head_tensor.assign(score_head_buffer);

    get_softmax_kernel(device_type)(score_head_tensor, nullptr);

    tensor::Tensor output(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                          const_cast<float*>(mha_out.ptr<float>()) + h * head_size);

    float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
    if (device_type == base::DeviceType::kDeviceCPU) {
      std::memset(output_head_ptr, 0, sizeof(float) * head_size);
    } else {
      cudaMemset(output_head_ptr, 0, sizeof(float) * head_size);
    }
    tensor::Tensor output_tensor(base::DataType::kDataTypeFp32, head_size, false, nullptr,
                                 output_head_ptr);
    output_tensor.set_device_type(device_type);

    for (int32_t t = 0; t <= pos; t++) {
      const float score_value = *(score_head_addr + t);
      float* value_head_addr = const_cast<float*>(value_cache_tensor.ptr<float>()) +
                               layer_offset + t * kv_dim + (h / kv_mul) * head_size;
      tensor::Tensor value_tensor(base::DataType::kDataTypeFp32, head_size, false,
                                  nullptr, value_head_addr);
      value_tensor.set_device_type(device_type);
      get_scale_kernel(device_type)(score_value, value_tensor, stream);
      get_add_kernel(device_type)(output_tensor, value_tensor, output_tensor, stream);
    }
  }
}
}  // namespace kernel