#include "model/llama2.h"
#include <fcntl.h>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <sys/mman.h>
#include <utility>
#include "base/tick.h"
namespace model {
LLamaRawModelData::~LLamaRawModelData() {
  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, file_size);
    data = nullptr;
  }
  if (fd != -1) {
    close(fd);
    fd = -1;
  }
}

const float* LLamaRawModelData::weight(size_t offset) const {
  return weight_data + offset;
}

bool LLamaRawModelData::is_weight_valid(size_t peek) const {
  if (peek * sizeof(float) < file_size) {
    return true;
  } else {
    return false;
  }
}

LLama2Model::LLama2Model(std::string token_path, std::string model_path)
    : Model(base::ModelType::kModelTypeLLama2, std::move(token_path), std::move(model_path)) {
}

base::Status LLama2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }

  auto sentence_piece_processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  const auto& status = sentence_piece_processor->Load(token_path_);
  if (!status.ok()) {
    return error::PathNotValid(token_path_);
  }

  vocab_size_ = sentence_piece_processor->GetPieceSize();
  if (vocab_size_ <= 0) {
    return error::ModelParseError("The vocab size param read error from the model file!");
  }
  device_type_ = device_type;
  encode_layer_ =
      std::make_unique<op::EncodeLayer>(true, false, std::move(sentence_piece_processor));

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }

  init_mem();
  return error::Success();
}

base::Status LLama2Model::forward(const std::vector<int>& tokens, int32_t total_steps) {
  CHECK(device_type_ == base::DeviceType::kDeviceCPU);
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  StatusCheck(embedding_layer_->forward_i2o1(input_tokens, input_token_num, input_embeddings));

  int32_t pos = 0;
  int32_t next = -1;
  int32_t eos = this->encode_layer_->eos();
  tensor::Tensor pos_tensor = this->get_buffer(ModelBufferType::kInputPos);

  while (pos < total_steps) {
    // set input
    pos_tensor.index<int32_t>(0) = pos;
    // set pos
    tensor::Tensor input(base::DataType::kDataTypeFp32, dim_);
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
      StatusCheck(embedding_layer_->forward_i2o1(input_tokens, input_token_num, input_embeddings));

      std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
          dim_ * sizeof(float), nullptr, input_embeddings.ptr<float>(0), true);
      input.assign(input_emb_buffer);
    }

    input.set_device_type(device_type_);
    for (int32_t layer_idx = 0; layer_idx < layer_num_; ++layer_idx) {
      // attn rmsnorm
      auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
      StatusCheck(rmsnorm_layers_.at(layer_idx)->forward_i1o1(input, rmsnorm_output));

      // kv cache
      tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
      // wq wk wv @ input
      const auto& [key, val] = slice_kv_cache(layer_idx, pos);
      // query
      StatusCheck(wq_layers_.at(layer_idx)->forward_i1o1(
          get_buffer(ModelBufferType::kOutputRMSNorm), query));

      // key
      StatusCheck(wk_layers_.at(layer_idx)->forward_i1o1(rmsnorm_output, key));

      // value
      StatusCheck(wv_layers_.at(layer_idx)->forward_i1o1(rmsnorm_output, val));

      // rope
      StatusCheck(rope_layer_->forward_i3o1(query, key, pos_tensor, tensor::Tensor{}));

      // mha
      tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
      tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

      tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
      tensor::Tensor key_storage = get_buffer(ModelBufferType::kKeyStorage);
      tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);

      mha_layer_->set_pos(pos);
      mha_layer_->set_layer_index(layer_idx);
      StatusCheck(mha_layer_->forward_i5o1(query, score_storage, key_cache, val_cache, key_storage,
                                           mha_output));

      // wo @ attention output
      tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
      StatusCheck(wo_layers_.at(layer_idx)->forward_i1o1(mha_output, attn_output));

      // residual add
      StatusCheck(add_layer_->forward_i2o1(input, attn_output, input));

      // ffn rmsnorm
      tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
      StatusCheck(rmsnorm_layers_.at(layer_idx + layer_num_)->forward_i1o1(input, ffn_norm_output));

      // w1
      tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
      StatusCheck(w1_layers_.at(layer_idx)->forward_i1o1(ffn_norm_output, w1_output));

      // w3
      tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
      StatusCheck(w3_layers_.at(layer_idx)->forward_i1o1(ffn_norm_output, w3_ouput));

      // SwiGLU
      StatusCheck(swiglu_layer_->forward_i2o1(w1_output, w3_ouput, w1_output));

      // w2
      tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
      StatusCheck(w2_layers_.at(layer_idx)->forward_i1o1(w1_output, w2_output));

      // residual add
      StatusCheck(add_layer_->forward_i2o1(input, w2_output, input));
    }
    StatusCheck(rmsnorm_layers_.at(2 * layer_num_)->forward_i1o1(input, input));
    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    StatusCheck(cls_layer_->forward_i1o1(input, forward_output));

    const float* forward_output_ptr = forward_output.ptr<float>();
    if (pos < tokens.size() - 1) {
      next = tokens[pos + 1];
    } else {
      next = (int32_t)std::distance(
          forward_output_ptr,
          std::max_element(forward_output_ptr, forward_output_ptr + forward_output.size()));
    }

    std::string output_str = this->encode_layer_->decode(next);
    std::cout << output_str << " " << std::flush;
    if (next == eos) {
      break;
    }
    pos += 1;
  }
  std::cout << "\n";
  return base::error::Success();
}

base::Status LLama2Model::gen_model_from_file() {
  using namespace base;
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }
  auto config = std::make_unique<LlamaModelConfig>();
  if (fread(config.get(), sizeof(LlamaModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model file.");
  }

  dim_ = config->dim;
  hidden_dim_ = config->hidden_dim;
  layer_num_ = config->layer_num;
  head_num_ = config->head_num;
  kv_head_num_ = config->head_num;
  seq_len_ = config->seq_len;

  kv_dim_ = (config->dim * config->kv_head_num) / config->head_num;
  kv_mul_ = config->head_num / config->kv_head_num;
  head_size_ = config->dim / config->head_num;

  if (std::abs(config->vocab_size) != vocab_size_) {
    return error::ModelParseError(
        "Vocabulary size mismatch between the model file and the token list.");
  }

  raw_model_data_ = std::make_unique<LLamaRawModelData>();
  fseek(file, 0, SEEK_END);
  raw_model_data_->file_size = ftell(file);
  fclose(file);

  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  raw_model_data_->fd = fd;
  raw_model_data_->data = static_cast<float*>(
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0));

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }

  raw_model_data_->weight_data = raw_model_data_->data + sizeof(LlamaModelConfig) / sizeof(float);
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }

  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    return layer_create_status;
  }
  return error::Success();
}

std::vector<int32_t> LLama2Model::encode(const std::string& sentence) {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

void LLama2Model::create_embedding_layer() {
  embedding_layer_ = std::make_unique<op::EmbeddingLayer>(dim_, seq_len_, std::abs(vocab_size_));

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
    auto wq = std::make_unique<op::MatmulLayer>(dim, dim);
    wq->reset_input_size(1);
    wq->reset_output_size(1);
    wq->reset_weight_size(1);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos));
    wq->get_weight(0).set_device_type(device_type_);
    pos += dim * dim;
    wq_layers_.push_back(std::move(wq));
  }

  // create weight matrix for key
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto wk = std::make_unique<op::MatmulLayer>(kv_dim_, dim);
    wk->reset_input_size(1);
    wk->reset_output_size(1);
    wk->reset_weight_size(1);

    wk->set_weight(0, {kv_dim_, dim}, this->raw_model_data_->weight(pos));
    wk->get_weight(0).set_device_type(device_type_);
    wk_layers_.push_back(std::move(wk));
    pos += kv_dim_ * dim;
  }

  // create weight matrix for value
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto wv = std::make_unique<op::MatmulLayer>(kv_dim_, dim);
    wv->reset_input_size(1);
    wv->reset_output_size(1);
    wv->reset_weight_size(1);
    wv->set_weight(0, {kv_dim_, dim}, this->raw_model_data_->weight(pos));
    wv->get_weight(0).set_device_type(device_type_);
    wv_layers_.push_back(std::move(wv));
    pos += kv_dim_ * dim;
  }

  // create weight matrix for output
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto wo = std::make_unique<op::MatmulLayer>(dim, dim);
    wo->reset_input_size(1);
    wo->reset_output_size(1);
    wo->reset_weight_size(1);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos));
    wo->get_weight(0).set_device_type(device_type_);
    wo_layers_.push_back(std::move(wo));
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = hidden_dim_;
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto w1 = std::make_unique<op::MatmulLayer>(hidden_dim, dim);
    w1->reset_input_size(1);
    w1->reset_output_size(1);
    w1->reset_weight_size(1);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos));
    w1->get_weight(0).set_device_type(device_type_);
    w1_layers_.push_back(std::move(w1));
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto w2 = std::make_unique<op::MatmulLayer>(dim, hidden_dim);
    w2->reset_input_size(1);
    w2->reset_output_size(1);
    w2->reset_weight_size(1);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos));
    w2->get_weight(0).set_device_type(device_type_);
    w2_layers_.push_back(std::move(w2));
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < layer_num_; ++i) {
    auto w3 = std::make_unique<op::MatmulLayer>(hidden_dim, dim);
    w3->reset_input_size(1);
    w3->reset_output_size(1);
    w3->reset_weight_size(1);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos));
    w3->get_weight(0).set_device_type(device_type_);
    w3_layers_.push_back(std::move(w3));
    pos += dim * hidden_dim;
  }

  pos += dim;
  pos += seq_len_ * head_size_;
  cls_layer_ = std::make_unique<op::MatmulLayer>(vocab_size_, dim);
  cls_layer_->reset_input_size(1);
  cls_layer_->reset_output_size(1);
  cls_layer_->reset_weight_size(1);
  cls_layer_->set_weight(0, {vocab_size_, dim}, this->raw_model_data_->weight(pos));
  cls_layer_->get_weight(0).set_device_type(device_type_);
}

void LLama2Model::create_rmsnorm_layers() {
  size_t rmsnorm_pos = dim_ * std::abs(vocab_size_);

  for (int32_t i = 0; i < layer_num_; ++i) {
    std::unique_ptr<op::RmsNormLayer> rms_norm_layer = std::make_unique<op::RmsNormLayer>(dim_);
    rms_norm_layer->reset_input_size(1);
    rms_norm_layer->reset_output_size(1);
    rms_norm_layer->reset_weight_size(1);

    const float* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {dim_}, weight_rmsnorm);
    rms_norm_layer->get_weight(0).set_device_type(device_type_);
    rmsnorm_layers_.push_back(std::move(rms_norm_layer));

    rmsnorm_pos += dim_;
  }

  rmsnorm_pos += layer_num_ * dim_ * dim_;
  rmsnorm_pos += layer_num_ * dim_ * (kv_head_num_ * head_size_);
  rmsnorm_pos += layer_num_ * dim_ * (kv_head_num_ * head_size_);
  rmsnorm_pos += layer_num_ * dim_ * dim_;

  for (int32_t i = 0; i < layer_num_; ++i) {
    std::unique_ptr<op::RmsNormLayer> rms_norm_layer = std::make_unique<op::RmsNormLayer>(dim_);
    rms_norm_layer->reset_input_size(1);
    rms_norm_layer->reset_output_size(1);
    rms_norm_layer->reset_weight_size(1);

    const float* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {dim_}, weight_rmsnorm);
    rms_norm_layer->get_weight(0).set_device_type(device_type_);
    rmsnorm_layers_.push_back(std::move(rms_norm_layer));

    rmsnorm_pos += dim_;
  }

  rmsnorm_pos += layer_num_ * hidden_dim_ * dim_;
  rmsnorm_pos += layer_num_ * hidden_dim_ * dim_;
  rmsnorm_pos += layer_num_ * hidden_dim_ * dim_;

  std::unique_ptr<op::RmsNormLayer> rms_final_layer = std::make_unique<op::RmsNormLayer>(dim_);
  rms_final_layer->reset_input_size(1);
  rms_final_layer->reset_output_size(1);
  rms_final_layer->reset_weight_size(1);

  const float* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {dim_}, weight_rmsnorm_final);
  rms_final_layer->get_weight(0).set_device_type(device_type_);
  rmsnorm_layers_.push_back(std::move(rms_final_layer));
}

void LLama2Model::init_mem() {
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  int32_t max_seq_len = seq_len_;
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, static_cast<int32_t>(max_seq_len));
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
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, layer_num_, seq_len_, kv_dim_);

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

base::Status LLama2Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& LLama2Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& LLama2Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

std::pair<tensor::Tensor, tensor::Tensor> LLama2Model::slice_kv_cache(int32_t layer_idx,
                                                                      int32_t token_pos) {
  int32_t layer_offset = layer_idx * seq_len_ * kv_dim_;
  int32_t cache_offset = static_cast<int32_t>(layer_offset + token_pos * kv_dim_);

  float* key_cache_ptr = get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset);
  float* val_cache_ptr = get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset);

  auto key_cache =
      std::make_shared<base::Buffer>(kv_dim_ * sizeof(float), nullptr, key_cache_ptr, true);
  auto val_cache =
      std::make_shared<base::Buffer>(kv_dim_ * sizeof(float), nullptr, val_cache_ptr, true);
  key_cache->set_device_type(device_type_);
  val_cache->set_device_type(device_type_);
  tensor::Tensor key(base::DataType::kDataTypeFp32, kv_dim_);
  tensor::Tensor val(base::DataType::kDataTypeFp32, kv_dim_);
  key.assign(key_cache);
  val.assign(val_cache);
  return {key, val};
}

void LLama2Model::create_rope_layer() {
  rope_layer_ = std::make_unique<op::RoPELayer>(dim_, kv_dim_, head_size_);
  rope_layer_->reset_input_size(3);
  rope_layer_->reset_output_size(1);
}

void LLama2Model::create_mha_layers() {
  mha_layer_ =
      std::make_unique<op::MultiHeadAttention>(kv_mul_, kv_dim_, seq_len_, head_num_, head_size_);
  mha_layer_->reset_input_size(5);
  mha_layer_->reset_output_size(1);
}

void LLama2Model::create_add_layer() {
  add_layer_ = std::make_unique<op::VecAddLayer>();
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
        "Create the matmul layer in the attention and ffn attention layers for the llama model "
        "failed.");
  }

  if (w1_layers_.size() != layer_num_ || w2_layers_.size() != layer_num_ ||
      w3_layers_.size() != layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model failed.");
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
  if (!mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  create_swiglu_layer();
  if (!swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

void LLama2Model::create_swiglu_layer() {
  swiglu_layer_ = std::make_unique<op::SwiGLULayer>(hidden_dim_);
  swiglu_layer_->reset_input_size(2);
  swiglu_layer_->reset_output_size(1);
}

}  // namespace model