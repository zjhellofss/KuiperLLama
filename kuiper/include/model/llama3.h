#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_
#include <base/cuda_config.h>
#include "model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"
namespace model {

struct LLama2Layers {
  std::shared_ptr<op::Layer> add_layer_;
  std::shared_ptr<op::Layer> rope_layer_;
  std::shared_ptr<op::Layer> swiglu_layer_;
  std::shared_ptr<op::Layer> mha_layer_;

  std::vector<std::shared_ptr<op::Layer>> wq_layers_;
  std::vector<std::shared_ptr<op::Layer>> wk_layers_;
  std::vector<std::shared_ptr<op::Layer>> wv_layers_;
  std::vector<std::shared_ptr<op::Layer>> wo_layers_;

  std::vector<std::shared_ptr<op::Layer>> w1_layers_;
  std::vector<std::shared_ptr<op::Layer>> w2_layers_;
  std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<op::Layer>> w3_layers_;
  std::shared_ptr<op::Layer> cls_layer_;

  std::shared_ptr<op::Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class LLama2Model : public Model {
 public:
  explicit LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model);

  base::Status init(base::DeviceType device_type) override;

  base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       bool is_prompt, int& next) const override;

  base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                       int& next) const override;

  // Unified forward interface - supports all scenarios
  base::Status forward_unified(
      const tensor::Tensor& hidden_states,  // [num_tokens, dim]
      const base::BatchMetadata& batch_meta,
      std::vector<int32_t>& next_tokens) const override;

  op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

  // Get cuda config for external use
  std::shared_ptr<kernel::CudaConfig> get_cuda_config() const { return cuda_config_; }

  // Initialize batch buffers for multi-sequence processing
  void init_batch_buffers(int32_t num_seqs);

  // Batch forward for multi-sequence decode (legacy, calls forward_unified internally)
  // inputs: [num_seqs, dim] - one token per sequence
  // positions: [num_seqs] - position for each sequence
  // outputs: [num_seqs] - next token for each sequence
  base::Status forward_batch(const tensor::Tensor& inputs,
                             const tensor::Tensor& positions,
                             std::vector<int32_t>& next_tokens) const;

 private:
  void init_mem() override;

  base::Status create_layers() override;

  void create_param_layers() override;

  void create_nonparam_layers() override;

  void create_param_quant_layers() override;

  void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  // Paged attention MHA for decode phase
  void attention_mha_paged(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

  void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

  void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

  void cls_logits(const tensor::Tensor& input) const;

  int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

  // Helper methods for unified forward
  void ensure_unified_buffers(int32_t num_tokens) const;

 private:
  std::shared_ptr<kernel::CudaConfig> cuda_config_;
  std::unique_ptr<LLama2Layers> llama_layers_;
};
}  // namespace model

#endif