#include "model/llama2.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "base/tick.h"
#include "sampler/mult_sampler.h"
namespace model {

void LLama2Layers::to_cuda() {
  if (add_layer_) {
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    embedding_layer_->to_cuda();
  }

  for (auto& mha_layer : mha_layers_) {
    if (mha_layer) {
      mha_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->to_cuda();
    }
  }
  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->to_cuda();
    }
  }
  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->to_cuda();
    }
  }
}

LLama2Model::LLama2Model(std::string token_path, std::string model_path)
    : Model(base::ModelType::kModelTypeLLama2, std::move(token_path), std::move(model_path)) {}

base::Status LLama2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }

  device_type_ = device_type;
  if (device_type == base::DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cublasCreate(&cuda_config_->handle);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  init_mem();
  sampler_ = std::make_unique<sampler::ArgmaxSampler>();
  return error::Success();
}

base::Status LLama2Model::forward(const std::vector<int>& tokens, int32_t total_steps) {
  if (tokens.empty()) {
    return base::error::InvalidArgument("The token array is empty.");
  }

  const auto& embedding_output = embedding(tokens);
  int32_t pos = 0;
  int32_t next = -1;
  int32_t eos = encode_layer_->eos();
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  while (pos < total_steps && pos < config_->seq_len_) {
    // set input and pos
    pos_tensor.index<int32_t>(0) = pos;
    tensor::Tensor input(base::DataType::kDataTypeFp32, config_->dim_);
    fill_input(next, pos_tensor, tokens, input, embedding_output);

    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
      attention_rms(layer_idx, input);

      // attention (wq wk wv @ input)
      attention_qkv(layer_idx, pos_tensor);
      // multi-head attention
      attention_mha(layer_idx, pos_tensor);

      // feed forward
      feed_forward(layer_idx, input);
    }

    cls_logits(input);
    const std::string& decode_str = post_processing(pos, next, tokens);
    printf("%05d %s\n", next, decode_str.c_str());
    fflush(stdout);
    if (next == eos) {
      break;
    }
    pos += 1;
  }
  return base::error::Success();
}

void LLama2Model::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);
  llama_layers_->rope_layer_->set_cuda_config(cuda_config_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto mha_layer = std::make_shared<op::MultiHeadAttention>(
        device_type_, i, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
        config_->head_size_);
    mha_layer->set_cuda_config(cuda_config_);
    llama_layers_->mha_layers_.push_back(mha_layer);
  }

  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
  llama_layers_->add_layer_->set_cuda_config(cuda_config_);

  llama_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
  llama_layers_->swiglu_layer_->set_cuda_config(cuda_config_);
}

void LLama2Model::create_param_layers() {
  CHECK(llama_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  llama_layers_->embedding_layer_->set_cuda_config(cuda_config_);

  const float* weight_embedding = raw_model_data_->weight(0);
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                              weight_embedding, cpu_device_type);

  // create all matmul layer
  int32_t dim = config_->dim_;
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
  // create weight matrix for query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    wq->set_cuda_config(cuda_config_);
    llama_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;
  }

  // create weight matrix for key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    wk->set_cuda_config(cuda_config_);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    wv->set_cuda_config(cuda_config_);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    wo->set_cuda_config(cuda_config_);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += config_->layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    w1->set_cuda_config(cuda_config_);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    w2->set_cuda_config(cuda_config_);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    w3->set_cuda_config(cuda_config_);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  pos += dim;
  pos += config_->seq_len_ * config_->head_size_;

  llama_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(0), cpu_device_type);
  } else {
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(pos), cpu_device_type);
  }
  llama_layers_->cls_layer_->set_cuda_config(cuda_config_);

  // create rmsnorm layer
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const float* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    rms_norm_layer->set_cuda_config(cuda_config_);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const float* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    rms_norm_layer->set_cuda_config(cuda_config_);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const float* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  rms_final_layer->set_cuda_config(cuda_config_);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

std::vector<int32_t> LLama2Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

void LLama2Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    llama_layers_->to_cuda();
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();
  int32_t max_seq_len = config_->seq_len_;
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, static_cast<int32_t>(max_seq_len),
                              true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, max_seq_len, config_->dim_, true,
                                  alloc);

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

std::pair<tensor::Tensor, tensor::Tensor> LLama2Model::slice_kv_cache(int32_t layer_idx,
                                                                      int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

  float* key_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
  float* val_cache_ptr =
      const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

  auto key_cache = std::make_shared<base::Buffer>(config_->kv_dim_ * sizeof(float), nullptr,
                                                  key_cache_ptr, true);
  auto val_cache = std::make_shared<base::Buffer>(config_->kv_dim_ * sizeof(float), nullptr,
                                                  val_cache_ptr, true);
  key_cache->set_device_type(device_type_);
  val_cache->set_device_type(device_type_);
  tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_);
  tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_);
  key.assign(key_cache);
  val.assign(val_cache);
  return {key, val};
}

base::Status LLama2Model::create_layers() {
  using namespace base;
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLama2Layers>();
  }

  create_param_layers();
  create_nonparam_layers();

  if (!llama_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
      llama_layers_->wk_layers_.size() != config_->layer_num_ ||
      llama_layers_->wv_layers_.size() != config_->layer_num_ ||
      llama_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
        !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
      llama_layers_->w2_layers_.size() != config_->layer_num_ ||
      llama_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
        !llama_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!llama_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!llama_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (llama_layers_->mha_layers_.size() != config_->layer_num_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->mha_layers_.at(i)) {
      return error::InternalError("Create the mha layer for the llama model failed!");
    }
  }

  if (!llama_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  input_tokens.reshape({static_cast<int32_t>(tokens.size())});
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !llama_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(
      llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));
  op::EmbeddingOutput output;
  output.input_embeddings = input_embeddings;
  output.input_tokens = input_tokens;
  output.input_token_num = input_token_num;
  return output;
}

void LLama2Model::fill_input(int32_t next, const tensor::Tensor& pos_tensor,
                             const std::vector<int32_t>& tokens, tensor::Tensor& input,
                             const op::EmbeddingOutput& embedding_output) const {
  CHECK(llama_layers_ != nullptr);
  const int32_t pos = pos_tensor.index<int32_t>(0);
  if (pos < tokens.size()) {
    auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

    std::shared_ptr<base::Buffer> input_emb_buffer =
        std::make_shared<base::Buffer>(config_->dim_ * sizeof(float), nullptr,
                                       input_embeddings.ptr<float>(pos * config_->dim_), true);
    input.assign(input_emb_buffer);
  } else {
    CHECK(llama_layers_ != nullptr);
    // generate steps
    auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
    auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
    auto input_token_num = tensor::Tensor(base::DataType::kDataTypeInt32, 1);
    CHECK_NE(next, -1) << "The next token is -1.";
    input_tokens.reshape({1});
    input_tokens.index<int32_t>(0) = next;
    CHECK_NE(llama_layers_->embedding_layer_, nullptr)
        << "The embedding layer in the llama2 model is null pointer.";
    STATUS_CHECK(
        llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

    std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
        config_->dim_ * sizeof(float), nullptr, input_embeddings.ptr<float>(0), true);
    input.assign(input_emb_buffer);  // 1024
  }
  input.set_device_type(device_type_);
}

void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  // value
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(query, key, pos_tensor, tensor::Tensor{}));
}

void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  // VAL = [val1,val2,...val t]
  // output @ VAL = 最终的结果
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = llama_layers_->mha_layers_.at(layer_idx);
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}

void LLama2Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(llama_layers_->cls_layer_, nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

std::string LLama2Model::post_processing(int32_t pos, int32_t& next,
                                         const std::vector<int32_t>& tokens) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    cudaDeviceSynchronize();
    const float* forward_logits_cpu = get_buffer(ModelBufferType::kForwardOutputCPU).ptr<float>();
    cudaMemcpy(const_cast<float*>(forward_logits_cpu), forward_logits, forward_output.byte_size(),
               cudaMemcpyDeviceToHost);
    forward_logits = forward_logits_cpu;
  }

  if (pos < tokens.size() - 1) {
    next = tokens[pos + 1];
  } else {
    next = sampler_->sample(forward_logits, static_cast<int32_t>(forward_output.size()));
  }
  const std::string& output_str = this->encode_layer_->decode(next);
  return output_str;
}

}  // namespace model