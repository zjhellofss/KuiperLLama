#include "mha_kernel.h"
#include "../softmax_kernel_i.h"
namespace kernel {

void mha_kernel_cpu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                    int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                    const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                    const tensor::Tensor& score_tensor,
                    const tensor::Tensor& key_cache_tensor,
                    const tensor::Tensor& value_cache_tensor,
                    const tensor::Tensor& key_tensor) {
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  for (int32_t h = 0; h < head_num; ++h) {
    const float* score_head_addr = score_tensor.ptr<float>() + h * seq_len;
    const float* query_head_addr = query_tensor.ptr<float>() + h * head_size;

    const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    for (int32_t t = 0; t <= pos; t++) {
      const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset +
                                   t * kv_dim + (h / kv_mul) * head_size;

      // key tensor: pos + 1 , head_size
      allocator->memcpy(key_head_addr,
                        const_cast<float*>(key_tensor.ptr<float>(t * head_size)),
                        head_size * sizeof(float), base::MemcpyKind::kMemcpyCPU2CPU);
    }

    arma::fmat key_mat(const_cast<float*>(key_tensor.ptr<float>()), head_size, pos + 1,
                       false, true);
    arma::fmat query(const_cast<float*>(query_head_addr), 1, head_size, false, true);
    arma::fmat score(const_cast<float*>(score_head_addr), 1, pos + 1, false, true);
    score = (query * key_mat) / std::sqrt(static_cast<float>(head_size));

    auto score_head_buffer = std::make_shared<base::Buffer>(
        (pos + 1) * sizeof(float), nullptr, const_cast<float*>(score_head_addr), true);
    score_head_buffer->set_device_type(base::DeviceType::kDeviceCPU);
    tensor::Tensor score_head_tensor(base::DataType::kDataTypeFp32, pos + 1);
    score_head_tensor.assign(score_head_buffer);

    get_softmax_kernel(base::DeviceType::kDeviceCPU)(score_head_tensor, nullptr);
    arma::fvec output_head(const_cast<float*>(mha_out.ptr<float>()) + h * head_size,
                           head_size, false, true);
    for (int32_t t = 0; t <= pos; t++) {
      const float score_value = score.at(t);
      const float* value_head_addr = value_cache_tensor.ptr<float>() + layer_offset +
                                     t * kv_dim + (h / kv_mul) * head_size;
      arma::fvec value(const_cast<float*>(value_head_addr), head_size, false, true);
      if (!t) {
        output_head = score_value * value;
      } else {
        output_head += score_value * value;
      }
    }
  }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return mha_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get an mha kernel.";
    return nullptr;
  }
}
}  // namespace kernel