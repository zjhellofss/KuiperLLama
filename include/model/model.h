#ifndef LC_INCLUDE_MODEL_MODEL_H_
#define LC_INCLUDE_MODEL_MODEL_H_
#include <string>
#include "llama2_config.h"
#include "op/embedding_layer.h"
#include "op/encode_layer.h"
#include "op/layer.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"

namespace model {
class Model {
 public:
  explicit Model(base::ModelType model_type, std::string token_path, std::string model_path);

  virtual base::Status init() = 0;

  virtual tensor::Tensor forward(const std::vector<int>& tokens, int start_pos) = 0;

  base::ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

 private:
  virtual base::Status read_model_file() = 0;

  virtual op::EmbeddingLayer* create_embedding_layer() = 0;

  virtual std::vector<int32_t> encode(const std::string& sentence) = 0;

 protected:
  base::ModelType model_type_;
  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<op::EncodeLayer> encode_layer_;
  std::unique_ptr<op::EmbeddingLayer> embedding_layer_;
};
}  // namespace model
#endif  // LC_INCLUDE_MODEL_MODEL_H_
