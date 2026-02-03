#ifndef KUIPER_INCLUDE_MODEL_MODEL_H_
#define KUIPER_INCLUDE_MODEL_MODEL_H_
#include <op/embedding.h>
#include <map>
#include <string>
#include <mutex>
#include "base/base.h"
#include "base/batch_metadata.h"
#include "config.h"
#include "op/encode.h"
#include "op/layer.h"
#include "raw_model_data.h"
#include "sampler/argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor/tensor.h"
#include "base/kv_cache_manager.h"
#include "base/block_allocator.h"
namespace model {
class Model {
 public:
  explicit Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                 std::string token_path, std::string model_path, bool is_quant_model);

  virtual base::Status init(base::DeviceType device_type) = 0;

  // Legacy single-token interface (deprecated, calls forward_unified internally)
  virtual base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               bool is_prompt, int& next) const = 0;

  // Legacy single-token forward (deprecated, calls forward_unified internally)
  virtual base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                               int& next) const = 0;

  // ============================================================================
  // Unified forward interface - supports all scenarios:
  // - Single token decode
  // - Batch decode (multiple sequences, one token each)
  // - Chunked prefill (single sequence, multiple tokens)
  // - Mixed prefill + decode (continuous batching)
  // ============================================================================
  virtual base::Status forward_unified(
      const tensor::Tensor& hidden_states,  // [num_tokens, dim]
      const base::BatchMetadata& batch_meta,
      std::vector<int32_t>& next_tokens) const;

  // Convenience methods
  base::Status forward_decode(
      const std::vector<int32_t>& seq_ids,
      const std::vector<int32_t>& positions,
      const tensor::Tensor& hidden_states,
      std::vector<int32_t>& next_tokens) const;

  base::Status forward_prefill(
      int32_t seq_id,
      const tensor::Tensor& hidden_states,  // [num_tokens, dim]
      int32_t start_pos,
      int32_t& next_token) const;

  // Chunked prefill: automatically splits long sequences into chunks
  base::Status forward_chunked_prefill(
      int32_t seq_id,
      const tensor::Tensor& hidden_states,  // [total_tokens, dim]
      int32_t start_pos,
      int32_t chunk_size,  // 0 = auto-compute
      int32_t& next_token) const;

  // Compute optimal chunk size based on available memory
  int32_t compute_chunk_size(int32_t total_tokens, int32_t num_active_seqs = 1) const;

  virtual base::Status init_paged_kv_cache(int32_t num_seqs, int32_t block_size,
                                           int32_t max_blocks,
                                           int32_t max_blocks_per_seq = -1);

  // Initialize unified buffers for variable token counts
  virtual void init_unified_buffers(int32_t max_tokens);

  base::ModelType model_type() const;

  const std::string& token_path() const;

  const std::string& model_path() const;

  virtual tensor::Tensor& get_buffer(ModelBufferType buffer_idx);

  virtual const tensor::Tensor& get_buffer(ModelBufferType buffer_idx) const;

  virtual bool is_sentence_ending(int32_t token_idx) const;

  virtual std::string decode(int32_t token_idx) const;

  virtual std::string decode(std::vector<int32_t> token_idxs) const;

  /////////////////////////////////////////////////////
  /////////////////////////////////////////////////////
  virtual std::vector<int32_t> encode(const std::string& sentence) const;

  virtual std::pair<tensor::Tensor, tensor::Tensor> slice_kv_cache(int32_t layer_idx,
                                                                   int32_t token_pos) const;

  virtual op::EmbeddingOutput embedding(const std::vector<int>& tokens) const = 0;

  virtual tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                                    const op::EmbeddingOutput& embedding_output,
                                    bool is_prompt) const;

  // Get model configuration
  const TransformerConfig* config() const { return config_.get(); }

  // Get KV cache manager (for paged attention)
  base::KVCacheManager* get_kv_cache_manager() const { return kv_cache_manager_.get(); }

 protected:
  virtual base::Status insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor);

  virtual base::Status read_model_file();

  virtual base::Status create_encode_layer();

  virtual base::Status gen_model_from_file();

  virtual base::Status generate_model_infos(const ModelConfig& config) const;

  virtual int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const = 0;

 private:
  virtual void init_mem() = 0;

  virtual base::Status create_layers() = 0;

  virtual void create_param_layers() = 0;

  virtual void create_nonparam_layers() = 0;

  virtual void create_param_quant_layers() = 0;

 protected:
  int32_t group_size_ = 1;
  int32_t num_seqs_ = 1;  // Number of sequences for batch processing
  int32_t max_unified_tokens_ = 0;  // Max tokens for unified buffers
  mutable std::mutex paged_kv_mutex_;
  std::unique_ptr<base::BlockAllocator> block_allocator_;
  std::unique_ptr<base::KVCacheManager> kv_cache_manager_;

  bool is_quant_model_ = false;
  std::unique_ptr<TransformerConfig> config_;

  std::string token_path_;
  std::string model_path_;
  std::unique_ptr<op::EncodeLayerBase> encode_layer_;
  std::map<ModelBufferType, tensor::Tensor> buffers_;
  std::unique_ptr<sampler::Sampler> sampler_;
  std::shared_ptr<RawModelData> raw_model_data_;
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
  base::ModelType model_type_ = base::ModelType::kModelTypeUnknown;
  base::TokenizerType tokenizer_type_ = base::TokenizerType::kEncodeUnknown;
};
}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_MODEL_H_
