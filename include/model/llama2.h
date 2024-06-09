#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <map>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
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

struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
};

class LLama2Model : public Model {
 public:
  explicit LLama2Model(std::string token_path, std::string model_path);

  base::Status init(base::DeviceType device_type) override;

  base::Status forward(const std::vector<int>& tokens, int32_t total_steps) override;

  std::vector<int32_t> encode(const std::string& sentence) override;

  std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                           int32_t token_pos) override;

 private:
  void init_mem() override;

  base::Status gen_model_from_file() override;

  base::Status read_model_file() override;

  base::Status create_layers() override;

  base::Status generate_llama_infos(const LLamaModelConfig& config);

  base::Status create_encode_layer();

  void create_rope_layer();

  void create_add_layer();

  void create_mha_layers();

  void create_rmsnorm_layers();

  void create_embedding_layer();

  void create_matmul_layers();

  void create_swiglu_layer();

  tensor::Tensor& get_buffer(ModelBufferType buffer_idx) override;

  const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const override;

  base::Status insert_buffer(ModelBufferType buffer_idx,
                             const tensor::Tensor& tensor) override;

  void attention_mha_o(int32_t layer_idx, int32_t pos);

  EmbeddingOutput prepare_input(const std::vector<int>& tokens);

  void attn_rmsnorm(const tensor::Tensor& input, int32_t layer_idx);

  void feed_forward(const tensor::Tensor& input, int32_t layer_idx);

  void fill_input(int32_t pos, int32_t next, const std::vector<int32_t>& tokens,
                  tensor::Tensor& input, const EmbeddingOutput& embedding_output);

  void attention_qkv(int32_t layer_idx, int32_t pos, const tensor::Tensor& pos_tensor);

 private:
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0;
  int32_t vocab_size_ = 0;

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;

  std::shared_ptr<LLamaRawModelData> raw_model_data_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<op::EncodeLayer> encode_layer_;

  std::shared_ptr<op::VecAddLayer> add_layer_;
  std::shared_ptr<op::RoPELayer> rope_layer_;
  std::shared_ptr<op::SwiGLULayer> swiglu_layer_;

  std::vector<std::shared_ptr<op::MultiHeadAttention>> mha_layers_;
  std::vector<std::shared_ptr<op::MatmulLayer>> wq_layers_;
  std::vector<std::shared_ptr<op::MatmulLayer>> wk_layers_;
  std::vector<std::shared_ptr<op::MatmulLayer>> wv_layers_;
  std::vector<std::shared_ptr<op::MatmulLayer>> wo_layers_;

  std::vector<std::shared_ptr<op::MatmulLayer>> w1_layers_;
  std::vector<std::shared_ptr<op::MatmulLayer>> w2_layers_;
  std::vector<std::shared_ptr<op::MatmulLayer>> w3_layers_;
  std::shared_ptr<op::MatmulLayer> cls_layer_;

  std::shared_ptr<op::EmbeddingLayer> embedding_layer_;
  std::vector<std::shared_ptr<op::RmsNormLayer>> rmsnorm_layers_;
};
}  // namespace model

#endif