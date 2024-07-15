#include "sampler/argmax_sampler.h"
#include <algorithm>
#include "../op/kernels/cuda/argmax_kernel.cuh"
namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    size_t next = std::distance(logits, std::max_element(logits, logits + size));
    return next;
  } else {
    size_t next = kernel::argmax_kernel_cu(logits, size, stream);
    return next;
  }
}
}  // namespace sampler