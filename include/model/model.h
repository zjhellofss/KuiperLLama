#ifndef LC_INCLUDE_MODEL_MODEL_H_
#define LC_INCLUDE_MODEL_MODEL_H_
#include <string>
#include "llama2_config.h"
#include "op/encode_layer.h"
#include "op/layer.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"

class Model {
 public:
  explicit Model(ModelType model_type, std::string token_path, std::string model_path);

  virtual Status init() = 0;

  virtual Tensor forward(const std::vector<int>& tokens, int start_pos) = 0;

  ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

 protected:
  ModelType model_type_;
  std::string token_path_;
  std::string model_path_;
  EncodeLayer encode_layer_;
};
#endif  // LC_INCLUDE_MODEL_MODEL_H_
