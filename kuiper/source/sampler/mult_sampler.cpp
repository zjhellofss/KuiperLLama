#include "sampler/mult_sampler.h"
#include "../op/kernels/softmax_kernel_i.h"
namespace sampler {
int32_t MultSampler::sample(const float* logits, int32_t size) {
  kernel::softmax_inplace_cpu(logits, size);
  float cdf = 0.0f;
  float coin = dist_(mt_);
  for (int i = 0; i < size; i++) {
    cdf += logits[i];
    if (coin < cdf) {
      return i;
    }
  }
  return size - 1;
}

MultSampler::MultSampler() {
  std::random_device rd;
  mt_ = std::mt19937(rd());
  dist_ = std::uniform_real_distribution<float>(0, 1);
}
}  // namespace sampler
