#include "model/model.h"
Model::Model(ModelType model_type, std::string token_path, std::string model_path)
    : model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)) {
}

ModelType Model::model_type() const {
  return model_type_;
}

const std::string& Model::token_path() const {
  return token_path_;
}

const std::string& Model::model_path() const {
  return model_path_;
}
