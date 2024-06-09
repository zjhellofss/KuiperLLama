#include "model/llama2.h"
#include <fcntl.h>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <sys/mman.h>
#include <array>
#include <utility>
#include "base/tick.h"
namespace model {

LLama2Model::LLama2Model(std::string token_path, std::string model_path)
    : Model(base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path)) {
}

base::Status LLama2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }

  device_type_ = device_type;
  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  init_mem();
  sampler_ = std::make_unique<sampler::ArgmaxSampler>();
  return error::Success();
}

base::Status LLama2Model::forward(const std::vector<int>& tokens, int32_t total_steps) {
  CHECK(device_type_ == base::DeviceType::kDeviceCPU);
  const auto& embedding_output = prepare_input(tokens);

  int32_t pos = 0;
  int32_t next = -1;
  int32_t eos = encode_layer_->eos();
  tensor::Tensor pos_tensor = get_buffer(ModelBufferType::kInputPos);
  TICK(A)
  while (pos < total_steps) {
    // set input and pos
    pos_tensor.index<int32_t>(0) = pos;
    tensor::Tensor input(base::DataType::kDataTypeFp32, dim_);
    fill_input(pos, next, tokens, input, embedding_output);

    for (int32_t layer_idx = 0; layer_idx < layer_num_; ++layer_idx) {
      attn_rmsnorm(input, layer_idx);

      // kv cache
      tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
      const auto& [key, val] = slice_kv_cache(layer_idx, pos);

      // attention (wq wk wv @ input)
      attention_qkv(layer_idx, pos, pos_tensor);
      attention_mha_o(layer_idx, pos);

      feed_forward(input, layer_idx);
    }
    STATUS_CHECK(rmsnorm_layers_.at(2 * layer_num_)->forward_i1o1(input, input));
    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    STATUS_CHECK(cls_layer_->forward_i1o1(input, forward_output));

    const float* forward_logist = forward_output.ptr<float>();
    if (pos < tokens.size() - 1) {
      next = tokens[pos + 1];
    } else {
      next = sampler_->sample(forward_logist, forward_output.size());
    }
    std::string output_str = this->encode_layer_->decode(next);
    std::cout << output_str << " " << std::flush;
    if (next == eos) {
      break;
    }
    pos += 1;
  }
  TOCK(A)
  std::cout << "word(pos) number: " << pos;
  return base::error::Success();
}

std::vector<int32_t> LLama2Model::encode(const std::string& sentence) {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

void LLama2Model::create_embedding_layer() {
  embedding_layer_ = std::make_shared<op::EmbeddingLayer>(device_type_, dim_, seq_len_,
                                                          std::abs(vocab_size_));

  const float* weight_embedding = raw_model_data_->weight(0);
  embedding_layer_->reset_weight_size(1);
  embedding_layer_->reset_input_size(2);
  embedding_layer_->reset_output_size(1);
  embedding_layer_->set_weight(0, {std::abs(vocab_size_), dim_}, weight_embedding);
  embedding_layer_->get_weight(0).set_device_type(device_type_);
}

void LLama2Model::create_matmul_layers() {
  int32_t dim = dim_;
  size_t pos = dim * std::abs(vocab_size_) + dim * layer_num_;
  // create weight matrix for query
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->reset_input_size(1);
    wq->reset_output_size(1);
    wq->reset_weight_size(1);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos));
    wq->get_weight(0).set_device_type(device_type_);
    pos += dim * dim;
    wq_layers_.push_back(wq);
  }

  // create weight matrix for key
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim_, dim);
    wk->reset_input_size(1);
    wk->reset_output_size(1);
    wk->reset_weight_size(1);

    wk->set_weight(0, {kv_dim_, dim}, this->raw_model_data_->weight(pos));
    wk->get_weight(0).set_device_type(device_type_);
    wk_layers_.push_back(wk);
    pos += kv_dim_ * dim;
  }

  // create weight matrix for value
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim_, dim);
    wv->reset_input_size(1);
    wv->reset_output_size(1);
    wv->reset_weight_size(1);
    wv->set_weight(0, {kv_dim_, dim}, this->raw_model_data_->weight(pos));
    wv->get_weight(0).set_device_type(device_type_);
    wv_layers_.push_back(wv);
    pos += kv_dim_ * dim;
  }

  // create weight matrix for output
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->reset_input_size(1);
    wo->reset_output_size(1);
    wo->reset_weight_size(1);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos));
    wo->get_weight(0).set_device_type(device_type_);
    wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = hidden_dim_;
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->reset_input_size(1);
    w1->reset_output_size(1);
    w1->reset_weight_size(1);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos));
    w1->get_weight(0).set_device_type(device_type_);
    w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->reset_input_size(1);
    w2->reset_output_size(1);
    w2->reset_weight_size(1);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos));
    w2->get_weight(0).set_device_type(device_type_);
    w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->reset_input_size(1);
    w3->reset_output_size(1);
    w3->reset_weight_size(1);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos));
    w3->get_weight(0).set_device_type(device_type_);
    w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  pos += dim;
  pos += seq_len_ * head_size_;

  cls_layer_ = std::make_shared<op::MatmulLayer>(device_type_, vocab_size_, dim);
  cls_layer_->reset_input_size(1);
  cls_layer_->reset_output_size(1);
  cls_layer_->reset_weight_size(1);
  if (is_shared_weight_) {
    // using token embedding weight
    cls_layer_->set_weight(0, {vocab_size_, dim}, this->raw_model_data_->weight(0));
  } else {
    cls_layer_->set_weight(0, {vocab_size_, dim}, this->raw_model_data_->weight(pos));
  }
  cls_layer_->get_weight(0).set_device_type(device_type_);
}

void LLama2Model::create_rmsnorm_layers() {
  size_t rmsnorm_pos = dim_ * std::abs(vocab_size_);

  for (int32_t i = 0; i < layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim_);
    rms_norm_layer->reset_input_size(1);
    rms_norm_layer->reset_output_size(1);
    rms_norm_layer->reset_weight_size(1);

    const float* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {dim_}, weight_rmsnorm);
    rms_norm_layer->get_weight(0).set_device_type(device_type_);
    rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += dim_;
  }

  rmsnorm_pos += layer_num_ * dim_ * dim_;
  rmsnorm_pos += layer_num_ * dim_ * (kv_head_num_ * head_size_);
  rmsnorm_pos += layer_num_ * dim_ * (kv_head_num_ * head_size_);
  rmsnorm_pos += layer_num_ * dim_ * dim_;

  for (int32_t i = 0; i < layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim_);
    rms_norm_layer->reset_input_size(1);
    rms_norm_layer->reset_output_size(1);
    rms_norm_layer->reset_weight_size(1);

    const float* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {dim_}, weight_rmsnorm);
    rms_norm_layer->get_weight(0).set_device_type(device_type_);
    rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += dim_;
  }

  rmsnorm_pos += layer_num_ * hidden_dim_ * dim_;
  rmsnorm_pos += layer_num_ * hidden_dim_ * dim_;
  rmsnorm_pos += layer_num_ * hidden_dim_ * dim_;

  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, dim_);
  rms_final_layer->reset_input_size(1);
  rms_final_layer->reset_output_size(1);
  rms_final_layer->reset_weight_size(1);

  const float* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {dim_}, weight_rmsnorm_final);
  rms_final_layer->get_weight(0).set_device_type(device_type_);
  rmsnorm_layers_.push_back(rms_final_layer);
}

void LLama2Model::init_mem() {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  int32_t max_seq_len = seq_len_;
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32,
                              static_cast<int32_t>(max_seq_len));
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, max_seq_len, dim_);

  input_tokens.allocate(alloc);
  input_embeddings.allocate(alloc);
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, dim_);
  rms_output.allocate(alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor score_storage(base::DataType::kDataTypeFp32, head_size_, seq_len_);
  score_storage.allocate(alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyStorage, score_storage));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, hidden_dim_);
  w1_output.allocate(alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, hidden_dim_);
  w3_output.allocate(alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, layer_num_, seq_len_, kv_dim_);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, layer_num_, seq_len_,
                             kv_dim_);

  key_cache.allocate(alloc);
  value_cache.allocate(alloc);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(base::DataType::kDataTypeFp32, dim_);
  query.allocate(alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1);
  pos_tensor.allocate(alloc);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  tensor::Tensor attn(base::DataType::kDataTypeFp32, head_num_, seq_len_);
  attn.allocate(alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, vocab_size_);
  forward_output.allocate(alloc);
  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

std::pair<tensor::Tensor, tensor::Tensor> LLama2Model::slice_kv_cache(int32_t layer_idx,
                                                                      int32_t token_pos) {
  int32_t layer_offset = layer_idx * seq_len_ * kv_dim_;
  int32_t cache_offset = static_cast<int32_t>(layer_offset + token_pos * kv_dim_);

  float* key_cache_ptr = get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset);
  float* val_cache_ptr =
      get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset);

  auto key_cache = std::make_shared<base::Buffer>(kv_dim_ * sizeof(float), nullptr,
                                                  key_cache_ptr, true);
  auto val_cache = std::make_shared<base::Buffer>(kv_dim_ * sizeof(float), nullptr,
                                                  val_cache_ptr, true);
  key_cache->set_device_type(device_type_);
  val_cache->set_device_type(device_type_);
  tensor::Tensor key(base::DataType::kDataTypeFp32, kv_dim_);
  tensor::Tensor val(base::DataType::kDataTypeFp32, kv_dim_);
  key.assign(key_cache);
  val.assign(val_cache);
  return {key, val};
}

void LLama2Model::create_rope_layer() {
  rope_layer_ = std::make_shared<op::RoPELayer>(device_type_, dim_, kv_dim_, head_size_);
  rope_layer_->reset_input_size(3);
  rope_layer_->reset_output_size(1);
}

void LLama2Model::create_mha_layers() {
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto mha_layer = std::make_shared<op::MultiHeadAttention>(
        device_type_, i, kv_mul_, kv_dim_, seq_len_, head_num_, head_size_);
    mha_layer->reset_input_size(5);
    mha_layer->reset_output_size(1);
    mha_layers_.push_back(mha_layer);
  }
}

void LLama2Model::create_add_layer() {
  add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
  add_layer_->reset_input_size(2);
  add_layer_->reset_output_size(1);
}

base::Status LLama2Model::create_layers() {
  using namespace base;

  create_embedding_layer();
  if (!embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  create_rmsnorm_layers();
  if (rmsnorm_layers_.size() != 2 * layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  create_matmul_layers();
  if (wq_layers_.size() != layer_num_ || wk_layers_.size() != layer_num_ ||
      wv_layers_.size() != layer_num_ || wo_layers_.size() != layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < layer_num_; ++i) {
    if (!wq_layers_.at(i) || !wk_layers_.at(i) || !wv_layers_.at(i) ||
        !wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (w1_layers_.size() != layer_num_ || w2_layers_.size() != layer_num_ ||
      w3_layers_.size() != layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < layer_num_; ++i) {
    if (!w1_layers_.at(i) || !w2_layers_.at(i) || !w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  create_rope_layer();
  if (!rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  create_add_layer();
  if (!add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  create_mha_layers();
  if (mha_layers_.size() != layer_num_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }
  for (int32_t i = 0; i < layer_num_; ++i) {
    if (!mha_layers_.at(i)) {
      return error::InternalError("Create the mha layer for the llama model failed!");
    }
  }

  create_swiglu_layer();
  if (!swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

void LLama2Model::create_swiglu_layer() {
  swiglu_layer_ = std::make_shared<op::SwiGLULayer>(device_type_, hidden_dim_);
  swiglu_layer_->reset_input_size(2);
  swiglu_layer_->reset_output_size(1);
}

EmbeddingOutput LLama2Model::prepare_input(const std::vector<int>& tokens) {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !embedding_layer_);
  STATUS_CHECK(
      embedding_layer_->forward_i2o1(input_tokens, input_token_num, input_embeddings));

  EmbeddingOutput output;
  output.input_embeddings = input_embeddings;
  output.input_tokens = input_tokens;
  output.input_token_num = input_token_num;
  return output;
}

void LLama2Model::fill_input(int32_t pos, int32_t next,
                             const std::vector<int32_t>& tokens, tensor::Tensor& input,
                             const EmbeddingOutput& embedding_output) {
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;
  if (pos < tokens.size()) {
    // prefill steps
    std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
        dim_ * sizeof(float), nullptr, input_embeddings.ptr<float>(pos * dim_), true);
    input.assign(input_emb_buffer);
  } else {
    // generate steps
    CHECK_NE(next, -1);
    input_token_num.reshape({1});
    input_tokens.index<int32_t>(0) = next;
    STATUS_CHECK(
        embedding_layer_->forward_i2o1(input_tokens, input_token_num, input_embeddings));

    std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
        dim_ * sizeof(float), nullptr, input_embeddings.ptr<float>(0), true);
    input.assign(input_emb_buffer);
  }
  input.set_device_type(device_type_);
}

void LLama2Model::attn_rmsnorm(const tensor::Tensor& input, int32_t layer_idx) {
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward_i1o1(input, rmsnorm_output));
}

void LLama2Model::attention_qkv(int32_t layer_idx, int32_t pos,
                                const tensor::Tensor& pos_tensor) {
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = wq_layers_.at(layer_idx);
  STATUS_CHECK(
      query_layer->forward_i1o1(get_buffer(ModelBufferType::kOutputRMSNorm), query));

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  // key
  const auto& key_layer = wk_layers_.at(layer_idx);
  STATUS_CHECK(key_layer->forward_i1o1(rmsnorm_output, key));

  // value
  const auto& value_layer = wv_layers_.at(layer_idx);
  STATUS_CHECK(value_layer->forward_i1o1(rmsnorm_output, val));

  // rope
  STATUS_CHECK(rope_layer_->forward_i3o1(query, key, pos_tensor, tensor::Tensor{}));
}

void LLama2Model::attention_mha_o(int32_t layer_idx, int32_t pos) {
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor key_storage = get_buffer(ModelBufferType::kKeyStorage);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);

  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  const auto& mha_layer = mha_layers_.at(layer_idx);
  mha_layer->set_pos(pos);
  STATUS_CHECK(mha_layer->forward_i5o1(query, score_storage, key_cache, val_cache,
                                       key_storage, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = wo_layers_.at(layer_idx);
  STATUS_CHECK(wo_layer->forward_i1o1(mha_output, attn_output));
}

void LLama2Model::feed_forward(const tensor::Tensor& input, int32_t layer_idx) {
  // residual add
  STATUS_CHECK(
      add_layer_->forward_i2o1(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  STATUS_CHECK(
      rmsnorm_layers_.at(layer_idx + layer_num_)->forward_i1o1(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  STATUS_CHECK(w1_layers_.at(layer_idx)->forward_i1o1(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  STATUS_CHECK(w3_layers_.at(layer_idx)->forward_i1o1(ffn_norm_output, w3_ouput));

  // SwiGLU
  STATUS_CHECK(swiglu_layer_->forward_i2o1(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  STATUS_CHECK(w2_layers_.at(layer_idx)->forward_i1o1(w1_output, w2_output));

  // residual add
  STATUS_CHECK(add_layer_->forward_i2o1(input, w2_output, input));
}

}  // namespace model