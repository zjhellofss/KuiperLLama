#include "op/layer.h"
#include <glog/logging.h>
#include <numeric>
#include <utility>

Layer::Layer(LayerType layer_type, DataType data_type, std::string layer_name)
    : layer_type_(layer_type), data_type_(data_type), layer_name_(std::move(layer_name)) {}

DataType Layer::data_type() const { return data_type_; }

LayerType Layer::layer_type() const { return layer_type_; }

ParamLayerFp32::ParamLayerFp32(LayerType layer_type, std::string layer_name)
    : Layer(layer_type, DataType::kDataTypeFp32, std::move(layer_name)) {}

LayerStatus ParamLayerFp32::Init() { return LayerStatus::kFunctionUnImplement; }

LayerStatus ParamLayerFp32::Forward() { return LayerStatus::kFunctionUnImplement; }

void ParamLayerFp32::set_input(int32_t idx, const Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  CHECK(input.data_type() == DataType::kDataTypeFp32);
  this->inputs_.at(idx) = input;
}

void ParamLayerFp32::set_output(int32_t idx, const Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  CHECK(output.data_type() == DataType::kDataTypeFp32);
  this->outputs_.at(idx) = output;
}

Tensor ParamLayerFp32::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

Tensor ParamLayerFp32::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void ParamLayerFp32::set_weight(int32_t idx, const Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == DataType::kDataTypeFp32);
  weights_.at(idx) = weight;
}

Tensor ParamLayerFp32::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

void ParamLayerFp32::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                const float* weight_ptr) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());

  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  std::shared_ptr<Buffer> buffer =
      std::make_shared<Buffer>(size, nullptr, (void*)(weight_ptr), true);

  Tensor weight(DataType::kDataTypeFp32, dims);
  CHECK(weight.assign(buffer));
  weights_.at(idx) = weight;
}

void ParamLayerFp32::reset_input_size(size_t size) { inputs_.resize(size); }

void ParamLayerFp32::reset_output_size(size_t size) { outputs_.resize(size); }

void ParamLayerFp32::reset_weight_size(size_t size) { weights_.resize(size); }
