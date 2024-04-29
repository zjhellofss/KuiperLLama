#include "model/model.h"
namespace model {
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
}  // namespace model