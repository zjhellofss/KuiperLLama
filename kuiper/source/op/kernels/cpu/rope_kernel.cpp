#include "rope_kernel.h"
namespace kernel {
void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size,
                     const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                     const tensor::Tensor& input_pos) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += 2) {
    int32_t head_dim = i % head_size;
    float freq = 1.0f / std::pow(10000.0f, static_cast<float>(head_dim) /
                                               static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = std::cos(val);
    float fci = std::sin(val);
    int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
    for (int32_t v = 0; v < rotn; v++) {
      float* vec = const_cast<float*>(
          v == 0 ? input_q.ptr<float>()
                 : input_k.ptr<float>());  // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rope_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}
}  // namespace kernel