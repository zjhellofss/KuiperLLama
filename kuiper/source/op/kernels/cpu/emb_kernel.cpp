#include "emb_kernel.h"
namespace kernel {

void emb_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, int32_t vocab_size) {
  const int32_t input_num = static_cast<int32_t>(input.size());
  const int32_t weight_dim = weight.get_dim(1);

  const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
  for (int32_t i = 0; i < input_num; ++i) {
    int32_t token = *input.ptr<int32_t>(i);
    if (token > vocab_size) {
      LOG(FATAL) << "Token index is greater than vocab size.";
    } else {
      void* dest_ptr = (void*)output.ptr<float>(i * weight_dim);
      const void* src_ptr = (void*)weight.ptr<float>(token * weight_dim);
      CHECK(src_ptr != nullptr);
      CHECK(dest_ptr != nullptr);
      allocator->memcpy(src_ptr, dest_ptr, weight_dim * sizeof(float),
                        base::MemcpyKind::kMemcpyCPU2CPU);
    }
  }
}

EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return emb_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get an embedding kernel.";
    return nullptr;
  }
}
}  // namespace kernel