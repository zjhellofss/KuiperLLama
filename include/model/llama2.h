#ifndef LC_INCLUDE_MODEL_LLAMA_H_
#define LC_INCLUDE_MODEL_LLAMA_H_
#include <map>
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

class LLama2Model : public Model {
 public:
  explicit LLama2Model(std::string token_path, std::string model_path);

  base::Status init(base::DeviceType device_type) override;

  base::Status forward(const std::vector<int>& tokens, int start_pos) override;

  std::vector<int32_t> encode(const std::string& sentence) override;

 private:
  void init_mem() override;

  base::Status gen_model_from_file() override;

  void create_rmsnorm_layer() override;

  void create_embedding_layer() override;

  tensor::Tensor& get_buffer(ModelBufferIdx buffer_idx) override;

  const tensor::Tensor& get_buffer(ModelBufferIdx buffer_idx) const override;

  base::Status insert_buffer(ModelBufferIdx buffer_idx, const tensor::Tensor& tensor) override;

 private:
  int32_t vocab_size_ = 0;
  std::unique_ptr<LlamaModelConfig> config_;
  std::unique_ptr<LLamaRawModelData> raw_model_data_;
};
}  // namespace model

#endif