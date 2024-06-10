#ifndef LLAMA_INFER_SAMPLER_H
#define LLAMA_INFER_SAMPLER_H
#include <cstddef>
#include <cstdint>
namespace sampler {
class Sampler {
 public:
  virtual int32_t sample(const float* logits, size_t size) = 0;
};
}  // namespace sampler
#endif  // LLAMA_INFER_SAMPLER_H
