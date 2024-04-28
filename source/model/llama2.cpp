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

LLamaModel::LLamaModel(std::string token_path, std::string model_path)
    : Model(base::ModelType::kModelTypeLLama2, std::move(token_path), std::move(model_path)) {
}

base::Status LLamaModel::init() {
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
  return read_model_file();
}

tensor::Tensor LLamaModel::forward(const std::vector<int>& tokens, int start_pos) {
  std::shared_ptr<base::DeviceAllocator> alloc = std::make_shared<base::CPUDeviceAllocator>();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  tensor::Tensor output_embeddings(base::DataType::kDataTypeFp32,
                                   static_cast<int32_t>(tokens.size()) * config_.dim);

  input_tokens.allocate(alloc);
  output_embeddings.allocate(alloc);

  int32_t* input_tokens_ptr = input_tokens.ptr<int32_t>();
  for (const int& token : tokens) {
    *input_tokens_ptr = token;
    input_tokens_ptr += 1;
  }
  CHECK(embedding_layer_ != nullptr) << "Create embedding layer failed in the init stage.";
  embedding_layer_->set_input(0, input_tokens);
  embedding_layer_->set_output(0, output_embeddings);
  embedding_layer_->forward();
  return tensor::Tensor{};
}

base::Status LLamaModel::read_model_file() {
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    LOG(ERROR) << "Failed to open the file. The path may be invalid.";
    return base::Status::kPathNotValid;
  }
  if (fread(&config_, sizeof(LlamaModelConfig), 1, file) != 1) {
    LOG(ERROR) << "Failed to retrieve the configuration information from the model file.";
    return base::Status::kParamReadError;
  }

  if (std::abs(config_.vocab_size) != vocab_size_) {
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

std::vector<int32_t> LLamaModel::encode(const std::string& sentence) {
  CHECK(encode_layer_ != nullptr);
  op::EncodeLayer* encode_layer_ptr = dynamic_cast<op::EncodeLayer*>(encode_layer_.get());
  CHECK(encode_layer_ptr != nullptr);
  return encode_layer_ptr->encode(sentence);
}

op::EmbeddingLayer* LLamaModel::create_embedding_layer() {
  op::EmbeddingLayer* embedding_layer = new op::EmbeddingLayer();
  const float* weight_embedding = raw_model_data_->weight(0);
  embedding_layer->reset_weight_size(1);
  embedding_layer->reset_input_size(1);
  embedding_layer->reset_output_size(1);
  embedding_layer->set_weight(0, {std::abs(vocab_size_), config_.dim}, weight_embedding);
  return embedding_layer;
}

}  // namespace model