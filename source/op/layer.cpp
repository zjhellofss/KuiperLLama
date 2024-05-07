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

const std::string& BaseLayer::get_layer_name() const {
  return layer_name_;
}

void BaseLayer::set_layer_name(const std::string& layer_name) {
  layer_name_ = layer_name;
}

Layer::Layer(LayerType layer_type, std::string layer_name)
    : BaseLayer(layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {
}

base::Status Layer::init() {
  return base::error::Success();
}

base::Status Layer::base_forward() {
  return base::error::FunctionNotImplement("");
}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  this->outputs_.at(idx) = output;
}

const tensor::Tensor& Layer::get_input(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_input(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, inputs_.size());
  return inputs_.at(idx);
}

tensor::Tensor& Layer::get_output(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

base::Status Layer::check() {
  return base::error::Success();
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size) {
  inputs_.resize(size);
}

void Layer::reset_output_size(size_t size) {
  outputs_.resize(size);
}

size_t Layer::input_size() const {
  return inputs_.size();
}

size_t Layer::output_size() const {
  return outputs_.size();
}

LayerFp32Param::LayerFp32Param(LayerType layer_type, std::string layer_name)
    : Layer(layer_type, std::move(layer_name)) {
}

void LayerFp32Param::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  weights_.at(idx) = weight;
}

const tensor::Tensor& LayerFp32Param::get_weight(int32_t idx) const {
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

size_t LayerFp32Param::weight_size() const {
  return weights_.size();
}

base::Status Layer::forward_i1o1(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->base_forward();
}

base::Status Layer::forward_i2o1(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_output(0, output1);
  return this->base_forward();
}

base::Status Layer::forward_i3o1(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_output(0, output1);
  return this->base_forward();
}

tensor::Tensor& LayerFp32Param::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

}  // namespace op