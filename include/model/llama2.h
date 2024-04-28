#ifndef LC_INCLUDE_MODEL_LLAMA_H_
#define LC_INCLUDE_MODEL_LLAMA_H_
#include "model.h"
#include "op/embedding_layer.h"
namespace model {

struct LLamaRawModelData {
  int32_t fd = -1;
  size_t file_size = 0;
  float* data = nullptr;
  float* weight_data = nullptr;

  ~LLamaRawModelData();

  const float* weight(size_t offset) const;

  bool is_weight_valid(size_t peek) const;
};

class LLamaModel : public Model {
 public:
  explicit LLamaModel(std::string token_path, std::string model_path);

  base::Status init() override;

  tensor::Tensor forward(const std::vector<int>& tokens, int start_pos) override;

  std::vector<int32_t> encode(const std::string& sentence) override;

 private:
  base::Status read_model_file() override;

  op::EmbeddingLayer* create_embedding_layer() override;

 private:
  int32_t vocab_size_ = 0;
  LlamaModelConfig config_;
  std::unique_ptr<LLamaRawModelData> raw_model_data_;
};
}  // namespace model

#endif