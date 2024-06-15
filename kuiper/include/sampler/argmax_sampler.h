//
// Created by fss on 24-6-9.
//

#ifndef LLAMA_INFER_NON_SAMPLER_H
#define LLAMA_INFER_NON_SAMPLER_H
#include "sampler.h"
namespace sampler {
class ArgmaxSampler : public Sampler {
 public:
  int32_t sample(const float* logits, int32_t size) override;
};
}  // namespace sampler
#endif  // LLAMA_INFER_NON_SAMPLER_H
