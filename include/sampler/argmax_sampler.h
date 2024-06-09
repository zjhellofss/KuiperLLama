//
// Created by fss on 24-6-9.
//

#ifndef LLAMA_INFER_NON_SAMPLER_H
#define LLAMA_INFER_NON_SAMPLER_H
#include <cstddef>
#include <cstdint>
namespace sampler {
class ArgmaxSampler {
 public:
  int32_t sample(const float* logits, size_t size);
};
}  // namespace sampler
#endif  // LLAMA_INFER_NON_SAMPLER_H
