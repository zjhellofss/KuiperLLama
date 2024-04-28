#include "op/layer.h"
#include <glog/logging.h>
#include <numeric>
#include <utility>

namespace op {
BaseLayer::BaseLayer(LayerType layer_type, base::DataType data_type, std::string layer_name)
    : layer_type_(layer_type), data_type_(data_type), layer_name_(std::move(layer_name)) {
}

base::DataType BaseLayer::data_type() const {
  return data_type_;
}

LayerType BaseLayer::layer_type() const {
  return layer_type_;
}

LayerFp32::LayerFp32(LayerType layer_type, std::string layer_name)
    : BaseLayer(layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {
}

base::Status LayerFp32::init() {
  return base::Status::kFunctionUnImplement;
}

base::Status LayerFp32::forward() {
  return base::Status::kFunctionUnImplement;
}

void LayerFp32::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void LayerFp32::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

tensor::Tensor LayerFp32::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor LayerFp32::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void LayerFp32::reset_input_size(size_t size) {
  inputs_.resize(size);
}

void LayerFp32::reset_output_size(size_t size) {
  outputs_.resize(size);
}

LayerFp32Param::LayerFp32Param(LayerType layer_type, std::string layer_name)
    : LayerFp32(layer_type, std::move(layer_name)) {
}

void LayerFp32Param::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  weights_.at(idx) = weight;
}

tensor::Tensor LayerFp32Param::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

void LayerFp32Param::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                const float* weight_ptr) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());

  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, (void*)(weight_ptr), true);

  tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
  CHECK(weight.assign(buffer));
  weights_.at(idx) = weight;
}

void LayerFp32Param::reset_weight_size(size_t size) {
  weights_.resize(size);
}

}  // namespace op