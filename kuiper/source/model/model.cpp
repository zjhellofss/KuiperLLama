#include "model/model.h"
#include <algorithm> 
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  // if (buffers_.count(buffer_idx) > 0) {
  //   return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  // }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model "
        "file.");
  }
  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError(
          "Failed to retrieve the group size information from the model "
          "file.");
    }
  }

  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }

  if (!is_quant_model_) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
  }
  raw_model_data_->file_size = sb.st_size;

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }
  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig);
  } else {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_);
  }
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
#if defined(QWEN3_SUPPORT)
  config_->immediate_dim_ = config.immediate_dim_;
#endif
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#ifdef LLAMA3_SUPPORT
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

base::Status Model::init_paged_kv_cache(int32_t num_seqs, int32_t block_size,
                                     int32_t max_blocks, int32_t max_blocks_per_seq) {
  CHECK(config_ != nullptr);
  CHECK_GT(num_seqs, 0);
  CHECK_GT(block_size, 0);
  CHECK_GT(max_blocks, 0);

  // Store num_seqs for batch processing
  num_seqs_ = num_seqs;

  if (max_blocks_per_seq <= 0) {
    max_blocks_per_seq = (config_->seq_len_ + block_size - 1) / block_size;
  }

  std::shared_ptr<base::DeviceAllocator> alloc_dev;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc_dev = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc_dev = base::CUDADeviceAllocatorFactory::get_instance();
  }
  std::shared_ptr<base::DeviceAllocator> alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  if (!kv_cache_manager_) {
    kv_cache_manager_ = std::make_unique<base::KVCacheManager>();
  }

  return kv_cache_manager_->init_paged_token_major(
      device_type_, alloc_dev, alloc_cpu, num_seqs, config_->layer_num_, config_->kv_head_num_,
      config_->head_size_, block_size, max_blocks, max_blocks_per_seq, nullptr);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  if (kv_cache_manager_) {
    return kv_cache_manager_->slice_kv(0, layer_idx, token_pos); // 这里默认只有一个seq
  }

  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     key_cache_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                     val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
#if defined(QWEN3_SUPPORT)
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      config_->hidden_dim_ * sizeof(float), nullptr,
      input_embeddings.ptr<float>(index * config_->hidden_dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->hidden_dim_);

#else
  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * config_->dim_), true);
  tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
#endif
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

// ============================================================================
// Unified forward interface implementation
// ============================================================================

base::Status Model::forward_unified(
    const tensor::Tensor& hidden_states,
    const base::BatchMetadata& batch_meta,
    std::vector<int32_t>& next_tokens) const {
  // Default implementation - subclasses should override
  return base::error::FunctionNotImplement("forward_unified not implemented in base Model class");
}

void Model::init_unified_buffers(int32_t max_tokens) {
  if (max_tokens <= max_unified_tokens_) {
    return;  // Already have enough capacity
  }

  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const int32_t hidden_dim = config_->hidden_dim_;

  // Allocate unified buffers for [max_tokens, dim/kv_dim/hidden_dim]
  tensor::Tensor unified_query(base::DataType::kDataTypeFp32, max_tokens, dim, true, alloc);
  tensor::Tensor unified_key(base::DataType::kDataTypeFp32, max_tokens, kv_dim, true, alloc);
  tensor::Tensor unified_value(base::DataType::kDataTypeFp32, max_tokens, kv_dim, true, alloc);
  tensor::Tensor unified_rms_output(base::DataType::kDataTypeFp32, max_tokens, dim, true, alloc);
  tensor::Tensor unified_mha_output(base::DataType::kDataTypeFp32, max_tokens, dim, true, alloc);
  tensor::Tensor unified_attn_output(base::DataType::kDataTypeFp32, max_tokens, dim, true, alloc);
  tensor::Tensor unified_ffn_rms(base::DataType::kDataTypeFp32, max_tokens, dim, true, alloc);
  tensor::Tensor unified_w1_output(base::DataType::kDataTypeFp32, max_tokens, hidden_dim, true, alloc);
  tensor::Tensor unified_w3_output(base::DataType::kDataTypeFp32, max_tokens, hidden_dim, true, alloc);
  tensor::Tensor unified_w2_output(base::DataType::kDataTypeFp32, max_tokens, dim, true, alloc);

  buffers_[ModelBufferType::kUnifiedQuery] = unified_query;
  buffers_[ModelBufferType::kUnifiedKey] = unified_key;
  buffers_[ModelBufferType::kUnifiedValue] = unified_value;
  buffers_[ModelBufferType::kUnifiedRMSNormOutput] = unified_rms_output;
  buffers_[ModelBufferType::kUnifiedMHAOutput] = unified_mha_output;
  buffers_[ModelBufferType::kUnifiedAttnOutput] = unified_attn_output;
  buffers_[ModelBufferType::kUnifiedFFNRMSNorm] = unified_ffn_rms;
  buffers_[ModelBufferType::kUnifiedW1Output] = unified_w1_output;
  buffers_[ModelBufferType::kUnifiedW3Output] = unified_w3_output;
  buffers_[ModelBufferType::kUnifiedW2Output] = unified_w2_output;

  max_unified_tokens_ = max_tokens;

  LOG(INFO) << "Initialized unified buffers for max " << max_tokens << " tokens";
}

base::Status Model::forward_decode(
    const std::vector<int32_t>& seq_ids,
    const std::vector<int32_t>& positions,
    const tensor::Tensor& hidden_states,
    std::vector<int32_t>& next_tokens) const {
  CHECK(kv_cache_manager_ != nullptr) << "KV cache manager not initialized";

  std::shared_ptr<base::DeviceAllocator> alloc_gpu;
  std::shared_ptr<base::DeviceAllocator> alloc_cpu;
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc_gpu = base::CPUDeviceAllocatorFactory::get_instance();
  }
  alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // Get context lengths from KV cache manager
  std::vector<int32_t> context_lens(seq_ids.size());
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    context_lens[i] = kv_cache_manager_->get_context_len(seq_ids[i]);
  }

  auto batch_meta = base::BatchMetadata::create_decode(
      seq_ids, positions, context_lens, alloc_gpu, alloc_cpu, nullptr);

  return forward_unified(hidden_states, batch_meta, next_tokens);
}

base::Status Model::forward_prefill(
    int32_t seq_id,
    const tensor::Tensor& hidden_states,
    int32_t start_pos,
    int32_t& next_token) const {
  CHECK(kv_cache_manager_ != nullptr) << "KV cache manager not initialized";

  std::shared_ptr<base::DeviceAllocator> alloc_gpu;
  std::shared_ptr<base::DeviceAllocator> alloc_cpu;
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc_gpu = base::CPUDeviceAllocatorFactory::get_instance();
  }
  alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // Calculate num_tokens from hidden_states shape
  int32_t num_tokens = static_cast<int32_t>(hidden_states.size() / config_->dim_);
  int32_t context_len = start_pos + num_tokens;

  auto batch_meta = base::BatchMetadata::create_prefill(
      seq_id, start_pos, num_tokens, context_len, alloc_gpu, alloc_cpu, nullptr);

  std::vector<int32_t> next_tokens;
  auto status = forward_unified(hidden_states, batch_meta, next_tokens);
  if (!status) {
    return status;
  }

  if (!next_tokens.empty()) {
    next_token = next_tokens[0];
  }
  return base::error::Success();
}

int32_t Model::compute_chunk_size(int32_t total_tokens, int32_t num_active_seqs) const {
  // Default chunk size limits
  constexpr int32_t kMinChunkSize = 16;
  constexpr int32_t kMaxChunkSize = 2048;
  constexpr int32_t kDefaultChunkSize = 512;

  if (total_tokens <= kMinChunkSize) {
    return total_tokens;
  }

  // Estimate memory per token (rough approximation)
  // Each token needs: Q, K, V, intermediate buffers
  // Q: dim, K: kv_dim, V: kv_dim, MHA output: dim, FFN intermediates: 2*hidden_dim
  size_t per_token_floats = config_->dim_ * 4 +      // Q, attn_out, rms_out, w2_out
                            config_->kv_dim_ * 2 +   // K, V
                            config_->hidden_dim_ * 2; // W1, W3 outputs
  size_t per_token_bytes = per_token_floats * sizeof(float);

  // Try to get available GPU memory (simplified - in practice would use cudaMemGetInfo)
  // For now, use a conservative estimate based on typical GPU memory
  size_t available_memory = 2ULL * 1024 * 1024 * 1024;  // Assume 2GB available for activations

  // Account for other sequences in the batch
  size_t memory_for_prefill = available_memory / (num_active_seqs > 0 ? num_active_seqs : 1);

  int32_t max_tokens_by_memory = static_cast<int32_t>(memory_for_prefill / per_token_bytes);

  // Clamp to reasonable range
  int32_t chunk_size = std::min({
      total_tokens,
      max_tokens_by_memory,
      kMaxChunkSize
  });

  chunk_size = std::max(chunk_size, kMinChunkSize);

  // Round down to multiple of 16 for better GPU efficiency
  chunk_size = (chunk_size / 16) * 16;
  if (chunk_size == 0) chunk_size = kMinChunkSize;

  return chunk_size;
}

base::Status Model::forward_chunked_prefill(
    int32_t seq_id,
    const tensor::Tensor& hidden_states,
    int32_t start_pos,
    int32_t chunk_size,
    int32_t& next_token) const {
  CHECK(kv_cache_manager_ != nullptr) << "KV cache manager not initialized";

  int32_t total_tokens = static_cast<int32_t>(hidden_states.size() / config_->dim_);

  if (total_tokens == 0) {
    return base::error::InvalidArgument("Empty hidden_states for chunked prefill");
  }

  // Auto-compute chunk size if not specified
  if (chunk_size <= 0) {
    chunk_size = compute_chunk_size(total_tokens, 1);
  }

  std::shared_ptr<base::DeviceAllocator> alloc_gpu;
  std::shared_ptr<base::DeviceAllocator> alloc_cpu;
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc_gpu = base::CPUDeviceAllocatorFactory::get_instance();
  }
  alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t current_pos = start_pos;
  int32_t tokens_processed = 0;

  while (tokens_processed < total_tokens) {
    int32_t tokens_this_chunk = std::min(chunk_size, total_tokens - tokens_processed);

    // Create a view into hidden_states for this chunk
    // hidden_states: [total_tokens, dim]
    // chunk: [tokens_this_chunk, dim]
    int32_t offset = tokens_processed * config_->dim_;
    tensor::Tensor chunk_hidden(base::DataType::kDataTypeFp32,
                                tokens_this_chunk, config_->dim_,
                                false, nullptr,
                                const_cast<float*>(hidden_states.ptr<float>(offset)));
    chunk_hidden.set_device_type(device_type_);

    // Allocate blocks for this chunk
    int32_t context_len_after_chunk = current_pos + tokens_this_chunk;
    if (!kv_cache_manager_->allocate_blocks_for_tokens(seq_id, context_len_after_chunk)) {
      return base::error::InternalError("Failed to allocate KV cache blocks for prefill chunk");
    }

    // Create batch metadata for this chunk
    auto batch_meta = base::BatchMetadata::create_prefill(
        seq_id, current_pos, tokens_this_chunk, context_len_after_chunk,
        alloc_gpu, alloc_cpu, nullptr);

    // Forward this chunk
    std::vector<int32_t> chunk_next_tokens;
    auto status = forward_unified(chunk_hidden, batch_meta, chunk_next_tokens);
    if (!status) {
      return status;
    }

    // Update position
    current_pos += tokens_this_chunk;
    tokens_processed += tokens_this_chunk;

    // Keep the last token prediction
    if (!chunk_next_tokens.empty()) {
      next_token = chunk_next_tokens[0];
    }
  }

  // Update context length in KV cache manager
  kv_cache_manager_->set_context_len(seq_id, current_pos);

  return base::error::Success();
}

}  // namespace model