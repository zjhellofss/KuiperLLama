#include "model/llama2.h"
#include <fcntl.h>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <sys/mman.h>
#include <utility>
#include "op/embedding_layer.h"

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
  if (token_path_.empty()) {
    return base::Status::kPathNotValid;
  }

  auto sentence_piece_processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  const auto& status = sentence_piece_processor->Load(token_path_);
  if (!status.ok()) {
    LOG(ERROR) << "The tokenize model load failed, may be the path " << token_path_
               << " is not valid!";
    return base::Status::kPathNotValid;
  }

  vocab_size_ = sentence_piece_processor->GetPieceSize();
  if (vocab_size_ <= 0) {
    return base::Status::kParamReadError;
  }

  encode_layer_ =
      std::make_unique<op::EncodeLayer>(true, false, std::move(sentence_piece_processor));
  base::Status read_status = read_model_file();
  if (read_status != base::Status::kSuccess) {
    LOG(ERROR) << "Create layers in the llama model failed.";
    return read_status;
  }

  device_type_ = device_type;
  init_mem();
  return base::Status::kSuccess;
}

tensor::Tensor LLama2Model::forward(const std::vector<int>& tokens, int start_pos) {
  auto input_tokens = get_buffer(ModelBufferIdx::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferIdx::kInputEmbeddings);
  int32_t* input_tokens_ptr = input_tokens.ptr<int32_t>();
  for (const int& token : tokens) {
    *input_tokens_ptr = token;
    input_tokens_ptr += 1;
  }
  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  CHECK(embedding_layer_ != nullptr) << "Create embedding layer failed in the init stage.";
  embedding_layer_->set_input(0, input_tokens);
  embedding_layer_->set_input(1, input_token_num);
  embedding_layer_->set_output(0, input_embeddings);
  embedding_layer_->forward();
  return tensor::Tensor{};
}

base::Status LLama2Model::read_model_file() {
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    LOG(ERROR) << "Failed to open the file. The path may be invalid.";
    return base::Status::kPathNotValid;
  }
  config_ = std::make_unique<LlamaModelConfig>();
  if (fread(config_.get(), sizeof(LlamaModelConfig), 1, file) != 1) {
    LOG(ERROR) << "Failed to retrieve the configuration information from the model file.";
    return base::Status::kParamReadError;
  }

  if (std::abs(config_->vocab_size) != vocab_size_) {
    LOG(ERROR) << "Vocabulary size mismatch between the model file and the token list.";
    return base::Status::kParamReadError;
  }

  raw_model_data_ = std::make_unique<LLamaRawModelData>();
  fseek(file, 0, SEEK_END);
  raw_model_data_->file_size = ftell(file);
  fclose(file);

  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    LOG(ERROR) << "Failed to open the weight file " << model_path_
               << " may be the path does not exist!";
    return base::Status::kPathNotValid;
  }

  raw_model_data_->fd = fd;
  raw_model_data_->data = static_cast<float*>(
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0));

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    LOG(ERROR) << "Failed to map the weight file " << model_path_ << " into memory.";
    return base::Status::kWeightReadError;
  }

  raw_model_data_->weight_data = raw_model_data_->data + sizeof(LlamaModelConfig) / sizeof(float);
  if (raw_model_data_ == nullptr) {
    LOG(ERROR) << "Failed to map the weight file " << model_path_
               << " into memory, the pointer to weight start address is null";
    return base::Status::kWeightReadError;
  }

  auto embedding_layer = create_embedding_layer();
  if (!embedding_layer) {
    return base::Status::kCreateLayerFailed;
  }
  embedding_layer_ = std::unique_ptr<op::EmbeddingLayer>(embedding_layer);
  return base::Status::kSuccess;
}

std::vector<int32_t> LLama2Model::encode(const std::string& sentence) {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

op::EmbeddingLayer* LLama2Model::create_embedding_layer() {
  op::EmbeddingLayer* embedding_layer = new op::EmbeddingLayer();
  const float* weight_embedding = raw_model_data_->weight(0);
  embedding_layer->reset_weight_size(1);
  embedding_layer->reset_input_size(2);
  embedding_layer->reset_output_size(1);
  embedding_layer->set_weight(0, {std::abs(vocab_size_), config_->dim}, weight_embedding);
  return embedding_layer;
}

void LLama2Model::init_mem() {
  if (!config_) {
    return;
  }
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  int32_t max_seq_len = config_->seq_len;
  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, static_cast<int32_t>(max_seq_len));
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, max_seq_len * config_->dim);

  input_tokens.allocate(alloc);
  input_embeddings.allocate(alloc);
  CHECK(insert_buffer(ModelBufferIdx::kInputTokens, input_tokens) == base::Status::kSuccess);
  CHECK(insert_buffer(ModelBufferIdx::kInputEmbeddings, input_embeddings) ==
        base::Status::kSuccess);
}

base::Status LLama2Model::insert_buffer(ModelBufferIdx buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::Status::kKeyValueHasExist;
  }
  buffers_.insert({buffer_idx, tensor});
  return base::Status::kSuccess;
}

tensor::Tensor LLama2Model::get_buffer(ModelBufferIdx buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

}  // namespace model