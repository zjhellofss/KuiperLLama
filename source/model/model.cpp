#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
namespace model {
RawModelData::~RawModelData() {
  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, file_size);
    data = nullptr;
  }
  if (fd != -1) {
    close(fd);
    fd = -1;
  }
}

const float* RawModelData::weight(size_t offset) const {
  return weight_data + offset;
}

bool RawModelData::is_weight_valid(size_t peek) const {
  if (peek * sizeof(float) < file_size) {
    return true;
  } else {
    return false;
  }
}

Model::Model(base::ModelType model_type, std::string token_path, std::string model_path)
    : model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)) {
}

base::ModelType Model::model_type() const {
  return model_type_;
}

const std::string& Model::token_path() const {
  return token_path_;
}

const std::string& Model::model_path() const {
  return model_path_;
}

base::Status Model::insert_buffer(ModelBufferType buffer_idx,
                                  const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) +
                                    " has exits in the buffers");
  }
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
    return error::PathNotValid(
        "Failed to open the weight file, the model path is empty!");
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

  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }

  raw_model_data_ = std::make_shared<RawModelData>();
  fseek(file, 0, SEEK_END);
  raw_model_data_->file_size = ftell(file);
  fclose(file);

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      static_cast<float*>(mmap(nullptr, raw_model_data_->file_size, PROT_READ,
                               MAP_PRIVATE, raw_model_data_->fd, 0));

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory.");
  }

  raw_model_data_->weight_data =
      raw_model_data_->data + sizeof(ModelConfig) / sizeof(float);
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError(
        "Failed to map the weight file " + model_path_ +
        " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;

  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  if (std::abs(config.vocab_size) != config_->vocab_size_) {
    return base::error::ModelParseError(
        "Vocabulary size mismatch between the model file and the token list.");
  }
  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe =
      std::make_unique<sentencepiece::SentencePieceProcessor>();
  const auto& status = spe->Load(token_path_);
  if (!status.ok()) {
    return error::PathNotValid(token_path_);
  }

  config_->vocab_size_ = spe->GetPieceSize();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }

  // create token encode decode layer
  encode_layer_ =
      std::make_unique<op::EncodeLayer>(device_type_, true, false, std::move(spe));
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init s entence piece processor
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }

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

}  // namespace model