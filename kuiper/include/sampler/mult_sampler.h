#ifndef LLAMA_INFER_MULT_SAMPLER_H
#define LLAMA_INFER_MULT_SAMPLER_H
#include "sampler.h"
#include <random>
namespace sampler {
class MultSampler : public Sampler {
 public:
  explicit MultSampler();

  int32_t sample(const float* logits, int32_t size) override;

 private:
  std::mt19937 mt_;
  std::uniform_real_distribution<float> dist_;
};
}  // namespace sampler
#endif  // LLAMA_INFER_MULT_SAMPLER_H
