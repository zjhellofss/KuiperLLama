#include "op/layer.h"
#include <glog/logging.h>
Layer::Layer(LayerType layer_type, std::string layer_name)
    : layer_type_(layer_type), layer_name_(std::move(layer_name)) {

}
LayerStatus LayerFp32::Init() {
  return LayerStatus::kFunctionUnImplement;
}

LayerStatus LayerFp32::Forward() {
  return LayerStatus::kFunctionUnImplement;
}

void LayerFp32::set_input(int32_t idx, const Tensor<float> &input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void LayerFp32::set_output(int32_t idx, const Tensor<float> &output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

Tensor<float> LayerFp32::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

Tensor<float> LayerFp32::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void LayerFp32::set_weight(int32_t idx, const Tensor<float> &weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  weights_.at(idx) = weight;
}

Tensor<float> LayerFp32::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}
