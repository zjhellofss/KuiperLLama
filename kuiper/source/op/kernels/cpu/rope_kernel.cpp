#include "rope_kernel.h"
namespace kernel {
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                     const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                     const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += 2) {
    int32_t head_dim = i % head_size;
    float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim);
    float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim);

    int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
    for (int32_t v = 0; v < rotn; v++) {
      float* vec =
          const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                    : input_k.ptr<float>());  // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}
}  // namespace kernel