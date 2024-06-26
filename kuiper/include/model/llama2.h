#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
};

struct LLama2Layers {
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
  std::vector<std::shared_ptr<op::RmsNormLayer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::MatmulLayer>> w3_layers_;
  std::shared_ptr<op::MatmulLayer> cls_layer_;

  std::shared_ptr<op::EmbeddingLayer> embedding_layer_;

  void to_cuda();
};

class LLama2Model : public Model {
 public:
  explicit LLama2Model(std::string token_path, std::string model_path);

  base::Status init(base::DeviceType device_type) override;

  base::Status forward(const std::vector<int>& tokens, int32_t total_steps) override;

  std::vector<int32_t> encode(const std::string& sentence) const override;

  std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(
      int32_t layer_idx, int32_t token_pos) const override;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  EmbeddingOutput embedding(const std::vector<int>& tokens) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void fill_input(int32_t next, const tensor::Tensor& pos_tensor,
                  const std::vector<int32_t>& tokens, tensor::Tensor& input,
                  const EmbeddingOutput& embedding_output) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  std::string post_processing(int32_t pos, int32_t& next,
                              const std::vector<int32_t>& tokens) const override;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<LLama2Layers> llama_layers_;
};
}  // namespace model

#endif