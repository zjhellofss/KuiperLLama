#include "sampler/argmax_sampler.h"
#include <algorithm>
namespace sampler {
int32_t ArgmaxSampler::sample(const float* logits, int32_t size) {
  int32_t next = (int32_t)std::distance(logits, std::max_element(logits, logits + size));
  return next;
}
}  // namespace sampler