#include "op/layer.h"
#include <glog/logging.h>
#include <numeric>
#include <utility>

BaseLayer::BaseLayer(LayerType layer_type, DataType data_type, std::string layer_name)
    : layer_type_(layer_type), data_type_(data_type), layer_name_(std::move(layer_name)) {
}

DataType BaseLayer::data_type() const {
  return data_type_;
}

LayerType BaseLayer::layer_type() const {
  return layer_type_;
}

LayerNoParam::LayerNoParam(LayerType layer_type, std::string layer_name)
    : BaseLayer(layer_type, DataType::kDataTypeFp32, std::move(layer_name)) {
}

Status LayerNoParam::init() {
  return Status::kFunctionUnImplement;
}

Status LayerNoParam::forward() {
  return Status::kFunctionUnImplement;
}

void LayerNoParam::set_input(int32_t idx, const Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  CHECK(input.data_type() == DataType::kDataTypeFp32);
  this->inputs_.at(idx) = input;
}

void LayerNoParam::set_output(int32_t idx, const Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  CHECK(output.data_type() == DataType::kDataTypeFp32);
  this->outputs_.at(idx) = output;
}

Tensor LayerNoParam::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

Tensor LayerNoParam::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void LayerNoParam::reset_input_size(size_t size) {
  inputs_.resize(size);
}

void LayerNoParam::reset_output_size(size_t size) {
  outputs_.resize(size);
}