#ifndef KUIPER_INCLUDE_MODEL_MODEL_H_
#define KUIPER_INCLUDE_MODEL_MODEL_H_
#include <map>
#include <string>
#include "llama2_config.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/encode.h"
#include "op/layer.h"
#include "op/matmul.h"
#include "op/mha.h"
#include "op/rmsnorm.h"
#include "op/rope.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"

namespace model {
enum class ModelBufferType {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  kInputPos = 6,
  kScoreStorage = 7,
  kOutputMHA = 8,
  kAttnOutput = 9,
  kW1Output = 10,
  kW2Output = 11,
  kW3Output = 12,
  kFFNRMSNorm = 13,
  kKeyStorage = 14,

  kForwardOutput = 15,
};

class Model {
 public:
  explicit Model(base::ModelType model_type, std::string token_path, std::string model_path);

  virtual base::Status init(base::DeviceType device_type) = 0;

  virtual base::Status forward(const std::vector<int>& tokens, int start_pos) = 0;

  base::ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

 private:
  virtual void init_mem() = 0;

  virtual base::Status gen_model_from_file() = 0;

  virtual base::Status create_layers() = 0;

  virtual base::Status read_model_file() = 0;

  virtual std::vector<int32_t> encode(const std::string& sentence) = 0;

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx) = 0;

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const = 0;

  virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) = 0;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                   int32_t token_pos) = 0;

 protected:
  bool is_shared_weight_ = false;
  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<sampler::ArgmaxSampler> sampler_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_MODEL_H_
