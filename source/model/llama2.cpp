#include "model/llama2.h"
#include <fcntl.h>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <sys/mman.h>
#include <utility>
#include "op/embedding_layer.h"
#include "op/matmul.h"

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

base::Status LLama2Model::forward(const std::vector<int>& tokens, int step_pos) {
  auto input_tokens = get_buffer(ModelBufferIdx::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferIdx::kInputEmbeddings);
  int32_t* input_tokens_ptr = input_tokens.ptr<int32_t>();
  if (!input_tokens_ptr) {
    return base::error::InternalError("Can't get the input token pointer in the forward function.");
  }
  for (const int& token : tokens) {
    *input_tokens_ptr = token;
    input_tokens_ptr += 1;
  }
  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  if (!embedding_layer_) {
    return base::error::InternalError("Create embedding layer failed in the init stage.");
  }

  embedding_layer_->set_input(0, input_tokens);
  embedding_layer_->set_input(1, input_token_num);
  embedding_layer_->set_output(0, input_embeddings);
  auto embedding_status = embedding_layer_->forward();
  if (!embedding_status) {
    embedding_status.set_err_msg("The embedding layer forward failed: " +
                                 embedding_status.get_err_msg());
    return embedding_status;
  }

  if (rmsnorm_layers_.size() != config_->layer_num) {
    return base::error::InternalError("The model has a wrong size of attn_rmsnorm layers!");
  }

  auto rms_output = get_buffer(ModelBufferIdx::kOutputRMSNorm);
  if (rms_output.is_empty()) {
    return base::error::InternalError("The rms output tensor is empty!");
  }

  int32_t dim = config_->dim;
  int32_t layer_num = config_->layer_num;
  int32_t seq_len = config_->seq_len;
  int32_t kv_dim = (config_->dim * config_->kv_head_num) / config_->head_num;

  for (int32_t i = 0; i < tokens.size(); ++i) {
    int32_t pos = step_pos + i;
    std::shared_ptr<base::Buffer> rms_buffer = std::make_shared<base::Buffer>(
        dim * sizeof(float), nullptr, input_embeddings.index<float>(i * dim), true);
    tensor::Tensor rms_input(base::DataType::kDataTypeFp32, dim);
    rms_input.assign(rms_buffer);
    rms_input.set_device_type(device_type_);

    for (int32_t l = 0; l < layer_num; ++l) {
      const auto& attn_norm_layer = rmsnorm_layers_.at(l);
      attn_norm_layer->set_input(0, rms_input);
      attn_norm_layer->set_output(0, rms_output);
      attn_norm_layer->forward();

      int32_t layer_offset = l * seq_len * kv_dim;
      float* key_cache_ptr = get_buffer(ModelBufferIdx::kKeyCache).ptr<float>();
      float* val_cache_ptr = get_buffer(ModelBufferIdx::kValueCache).ptr<float>();

      // tensor::Tensor key_cache(base::DataType::kDataTypeFp32, layer_num * seq_len * kv_dim);
      auto key_cache_buffer = std::make_shared<base::Buffer>(
          dim * sizeof(float), nullptr, key_cache_ptr + layer_offset + pos * kv_dim, true);
      auto val_cache_buffer = std::make_shared<base::Buffer>(
          dim * sizeof(float), nullptr, val_cache_ptr + layer_offset + pos * kv_dim, true);
      key_cache_buffer->set_device_type(device_type_);
      val_cache_buffer->set_device_type(device_type_);
      tensor::Tensor key_cache(base::DataType::kDataTypeFp32, dim);
      tensor::Tensor val_cache(base::DataType::kDataTypeFp32, dim);
      key_cache.assign(key_cache_buffer);
      val_cache.assign(val_cache_buffer);
    }
  }

  return base::error::Success();
}

base::Status LLama2Model::gen_model_from_file() {
  using namespace base;
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }
  config_ = std::make_unique<LlamaModelConfig>();
  if (fread(config_.get(), sizeof(LlamaModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model file.");
  }

  if (std::abs(config_->vocab_size) != vocab_size_) {
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

  create_embedding_layer();
  if (!embedding_layer_) {
    return error::InternalError("Create the embedding layer failed!");
  }

  create_rmsnorm_layers();
  if (rmsnorm_layers_.size() != config_->layer_num) {
    return error::InternalError("Create the rmsnorm layer failed!");
  }

  create_matmul_layers();
  if (wq_layers_.size() != config_->layer_num || wk_layers_.size() != config_->layer_num ||
      wv_layers_.size() != config_->layer_num) {
    return error::InternalError("Create the matmul layer in the attention layers failed.");
  }
  return error::Success();
}

std::vector<int32_t> LLama2Model::encode(const std::string& sentence) {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

void LLama2Model::create_embedding_layer() {
  embedding_layer_ = std::make_unique<op::EmbeddingLayer>(config_->dim, config_->seq_len,
                                                          std::abs(config_->vocab_size));

  const float* weight_embedding = raw_model_data_->weight(0);
  embedding_layer_->reset_weight_size(1);
  embedding_layer_->reset_input_size(2);
  embedding_layer_->reset_output_size(1);
  embedding_layer_->set_weight(0, {std::abs(vocab_size_), config_->dim}, weight_embedding);
  embedding_layer_->get_weight(0).set_device_type(device_type_);
}

void LLama2Model::create_matmul_layers() {
  int32_t layer_num = config_->layer_num;
  int32_t dim = config_->dim;
  int32_t kv_dim = (config_->dim * config_->kv_head_num) / config_->head_num;

  int32_t pos = config_->dim * std::abs(config_->vocab_size) + dim * layer_num;
  for (int32_t i = 0; i < layer_num; ++i) {
    auto wq = std::make_unique<op::MatmulLayer>(dim, dim);
    wq->reset_input_size(1);
    wq->reset_output_size(1);
    wq->reset_weight_size(1);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos));
    wq->get_weight(0).set_device_type(device_type_);
    pos += dim * dim;
    wq_layers_.push_back(std::move(wq));
  }

  for (int32_t i = 0; i < layer_num; ++i) {
    auto wk = std::make_unique<op::MatmulLayer>(kv_dim, dim);
    wk->reset_input_size(1);
    wk->reset_output_size(1);
    wk->reset_weight_size(1);

    wk->set_weight(0, {kv_dim, dim}, this->raw_model_data_->weight(pos));
    wk->get_weight(0).set_device_type(device_type_);
    wk_layers_.push_back(std::move(wk));
    pos += kv_dim * dim;
  }

  for (int32_t i = 0; i < layer_num; ++i) {
    auto wv = std::make_unique<op::MatmulLayer>(kv_dim, dim);
    wv->reset_input_size(1);
    wv->reset_output_size(1);
    wv->reset_weight_size(1);
    wv->set_weight(0, {kv_dim, dim}, this->raw_model_data_->weight(pos));
    wv->get_weight(0).set_device_type(device_type_);
    wv_layers_.push_back(std::move(wv));
    pos += kv_dim * dim;
  }
}

void LLama2Model::create_rmsnorm_layers() {
  int32_t layer_num = config_->layer_num;
  int32_t rmsnorm_pos = config_->dim * std::abs(config_->vocab_size);

  for (int32_t i = 0; i < layer_num; ++i) {
    std::unique_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_unique<op::RmsNormLayer>(config_->dim);
    rms_norm_layer->reset_input_size(1);
    rms_norm_layer->reset_output_size(1);
    rms_norm_layer->reset_weight_size(1);

    const float* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim}, weight_rmsnorm);
    rms_norm_layer->get_weight(0).set_device_type(device_type_);
    rmsnorm_layers_.push_back(std::move(rms_norm_layer));

    rmsnorm_pos += config_->dim;
  }
}

void LLama2Model::init_mem() {
  if (!config_) {
    return;
  }
  CHECK(device_type_ == base::DeviceType::kDeviceCPU);
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  int32_t max_seq_len = config_->seq_len;
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, static_cast<int32_t>(max_seq_len));
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, max_seq_len, config_->dim);

  input_tokens.allocate(alloc);
  input_embeddings.allocate(alloc);
  CHECK(insert_buffer(ModelBufferIdx::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferIdx::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim);
  rms_output.allocate(alloc);
  CHECK(insert_buffer(ModelBufferIdx::kOutputRMSNorm, rms_output));

  int32_t kv_dim = (config_->dim * config_->kv_head_num) / config_->head_num;
  int32_t layer_num = config_->layer_num;
  int32_t seq_len = config_->seq_len;

  // kv cache
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, layer_num * seq_len * kv_dim);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, layer_num * seq_len * kv_dim);
  key_cache.allocate(alloc);
  value_cache.allocate(alloc);
  CHECK(insert_buffer(ModelBufferIdx::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferIdx::kValueCache, value_cache));
}

base::Status LLama2Model::insert_buffer(ModelBufferIdx buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  buffers_.insert({buffer_idx, tensor});
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  return base::error::Success();
}

tensor::Tensor& LLama2Model::get_buffer(ModelBufferIdx buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& LLama2Model::get_buffer(ModelBufferIdx buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

}  // namespace model