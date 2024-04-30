#ifndef LC_INCLUDE_MODEL_MODEL_H_
#define LC_INCLUDE_MODEL_MODEL_H_
#include <map>
#include <string>
#include "llama2_config.h"
#include "op/embedding_layer.h"
#include "op/encode_layer.h"
#include "op/layer.h"
#include "op/rmsnorm_layer.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"

namespace model {
enum class ModelBufferIdx {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
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

  virtual void create_embedding_layer() = 0;

  virtual void create_rmsnorm_layer() = 0;

  virtual std::vector<int32_t> encode(const std::string& sentence) = 0;

  virtual tensor::Tensor& get_buffer(ModelBufferIdx buffer_idx) = 0;

  virtual const tensor::Tensor& get_buffer(ModelBufferIdx buffer_idx) const = 0;

  virtual base::Status insert_buffer(ModelBufferIdx buffer_idx, const tensor::Tensor& tensor) = 0;

 protected:
  std::string token_path_;
  std::string model_path_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  std::map<ModelBufferIdx, tensor::Tensor> buffers_;
  std::unique_ptr<op::EncodeLayer> encode_layer_;
  std::vector<std::unique_ptr<op::RmsNormLayer>> rmsnorm_layers_;
  std::unique_ptr<op::EmbeddingLayer> embedding_layer_;
};
}  // namespace model
#endif  // LC_INCLUDE_MODEL_MODEL_H_
