#include "model/llama2.h"
#include <fcntl.h>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <sys/mman.h>
#include <utility>

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

void LLamaRawModelData::add_offset(size_t offset) {
  current_offset += offset;
}

const float* LLamaRawModelData::weight() const {
  return weight_data + current_offset;
}

bool LLamaRawModelData::weight_is_valid(size_t peek) const {
  if ((current_offset + peek) * sizeof(float) < file_size) {
    return true;
  } else {
    return false;
  }
}

LLamaModel::LLamaModel(std::string token_path, std::string model_path)
    : Model(ModelType::kModelTypeLLama2, std::move(token_path), std::move(model_path)) {
}

Status LLamaModel::init() {
  if (token_path_.empty()) {
    return Status::kPathNotValid;
  }

  sentence_piece_processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
  const auto& status = sentence_piece_processor_->Load(token_path_);
  if (!status.ok()) {
    LOG(ERROR) << "The tokenize model load failed, may be the path " << token_path_
               << " is not valid!";
    return Status::kPathNotValid;
  }

  vocab_size_ = sentence_piece_processor_->GetPieceSize();
  if (vocab_size_ <= 0) {
    return Status::kParamReadError;
  }
  return read_model_file();
}

Tensor LLamaModel::forward(const std::vector<int>& tokens, int start_pos) {
  return Tensor();
}

Status LLamaModel::read_model_file() {
  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    LOG(ERROR) << "Failed to open the file. The path may be invalid.";
    return Status::kPathNotValid;
  }
  if (fread(&config_, sizeof(LlamaModelConfig), 1, file) != 1) {
    LOG(ERROR) << "Failed to retrieve the configuration information from the model file.";
    return Status::kParamReadError;
  }

  if (std::abs(config_.vocab_size) != vocab_size_) {
    LOG(ERROR) << "Vocabulary size mismatch between the model file and the token list.";
    return Status::kParamReadError;
  }

  raw_model_data_ = std::make_unique<LLamaRawModelData>();
  fseek(file, 0, SEEK_END);
  raw_model_data_->file_size = ftell(file);
  fclose(file);

  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    LOG(ERROR) << "Failed to open the weight file " << model_path_
               << " may be the path does not exist!";
    return Status::kPathNotValid;
  }

  raw_model_data_->fd = fd;
  raw_model_data_->data = static_cast<float*>(
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0));

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    LOG(ERROR) << "Failed to map the weight file " << model_path_ << " into memory.";
    return Status::kWeightReadError;
  }

  raw_model_data_->weight_data = raw_model_data_->data + sizeof(LlamaModelConfig) / sizeof(float);
  if (raw_model_data_ == nullptr) {
    LOG(ERROR) << "Failed to map the weight file " << model_path_
               << " into memory, the pointer to weight start address is null";
    return Status::kWeightReadError;
  }
  return Status::kSuccess;
}
