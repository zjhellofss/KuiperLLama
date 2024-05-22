#include "op/layer.h"
#include <glog/logging.h>
#include <numeric>
#include <utility>

namespace op {
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {
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
base::DeviceType BaseLayer::device_type() const {
  return device_type_;
}

void BaseLayer::set_device_type(base::DeviceType device_type) {
  device_type_ = device_type;
}

Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {
}

base::Status Layer::init() {
  return base::error::Success();
}

base::Status Layer::base_forward() {
  return base::error::FunctionNotImplement("");
}

base::Status Layer::check_inout_size(size_t expected_in_num, size_t expected_out_num) const {
  if (expected_in_num != this->input_size()) {
    return base::error::InternalError("The input tensors in the layer is wrong");
  }
  if (expected_out_num != this->output_size()) {
    return base::error::InternalError("The output tensors in the layer is wrong");
  }
  return base::error::Success();
}

base::Status Layer::check_single_output(size_t out_idx, base::DeviceType device_type,
                                        base::DataType data_type) const {
  if (this->get_output(out_idx).is_empty()) {
    return base::error::InternalError("The output tensor " + std::to_string(out_idx) +
                                      " is empty.");
  }
  if (this->get_output(out_idx).data_type() != data_type) {
    return base::error::InternalError("The output tensor " + std::to_string(out_idx) +
                                      " has a wrong data type.");
  }
  if (this->get_output(out_idx).device_type() != device_type) {
    return base::error::InternalError("The output tensor " + std::to_string(out_idx) +
                                      " has a wrong device type.");
  }
  return base::error::Success();
}

base::Status Layer::check_single_input(size_t in_idx, base::DeviceType device_type,
                                       base::DataType data_type) const {
  if (this->get_input(in_idx).is_empty()) {
    return base::error::InternalError("The input tensor " + std::to_string(in_idx) + " is empty.");
  }
  if (this->get_input(in_idx).data_type() != data_type) {
    return base::error::InternalError("The input tensor " + std::to_string(in_idx) +
                                      " has a wrong data type.");
  }
  if (this->get_input(in_idx).device_type() != device_type) {
    return base::error::InternalError("The input tensor " + std::to_string(in_idx) +
                                      " has a wrong device type.");
  }
  return base::error::Success();
}

base::Status Layer::check_inout(size_t in_num, size_t out_num, base::DeviceType device_type,
                                base::DataType data_type) const {
  if (this->input_size() != in_num) {
    return base::error::InternalError("The input number is not equal to " + std::to_string(in_num));
  }
  if (this->output_size() != out_num) {
    return base::error::InternalError("The output number is not equal to " +
                                      std::to_string(out_num));
  }

  for (int32_t i = 0; i < in_num; ++i) {
    tensor::Tensor input = this->get_input(i);
    if (input.is_empty()) {
      return base::error::InternalError("The input tensor " + std::to_string(i) + " is empty.");
    }
    if (input.device_type() != device_type) {
      return base::error::InternalError("The input tensor " + std::to_string(i) +
                                        " has a wrong device type.");
    }
    if (input.data_type() != data_type) {
      return base::error::InternalError("The input tensor " + std::to_string(i) +
                                        " has a wrong data type.");
    }
  }

  for (int32_t i = 0; i < out_num; ++i) {
    tensor::Tensor output = this->get_output(i);
    if (output.is_empty()) {
      return base::error::InternalError("The output tensor " + std::to_string(i) + " is empty.");
    }
    if (output.device_type() != device_type) {
      return base::error::InternalError("The output tensor " + std::to_string(i) +
                                        " has a wrong device type.");
    }
    if (output.data_type() != data_type) {
      return base::error::InternalError("The output tensor " + std::to_string(i) +
                                        " has a wrong data type.");
    }
  }
  return base::error::Success();
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

base::Status Layer::check() const {
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

LayerFp32Param::LayerFp32Param(base::DeviceType device_type, LayerType layer_type,
                               std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)) {
}

void LayerFp32Param::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
  weights_.at(idx) = weight;
}

base::Status LayerFp32Param::check_inout_wei_size(size_t expected_in_num, size_t expected_out_num,
                                                  size_t expected_wei_num) const {
  if (expected_in_num != this->input_size()) {
    return base::error::InternalError("The size of input tensors in the layer is wrong");
  }
  if (expected_out_num != this->output_size()) {
    return base::error::InternalError("The size of output tensors in the layer is wrong");
  }
  if (expected_wei_num != this->weight_size()) {
    return base::error::InternalError("The size of weight tensors in the layer is wrong");
  }
  return base::error::Success();
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

base::Status Layer::forward_i4o1(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_output(0, output1);
  return this->base_forward();
}

base::Status Layer::forward_i5o1(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);
  this->set_output(0, output1);
  return this->base_forward();
}

tensor::Tensor& LayerFp32Param::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

base::Status LayerFp32Param::check_weight(size_t wei_num, base::DeviceType device_type,
                                          base::DataType data_type) const {
  using namespace base;
  if (weight_size() != wei_num) {
    return error::InternalError("The weight num is not equal to " + std::to_string(wei_num));
  }

  for (int32_t i = 0; i < wei_num; ++i) {
    tensor::Tensor wei = this->get_weight(i);
    if (wei.is_empty()) {
      return base::error::InternalError("The weight tensor " + std::to_string(i) + " is empty.");
    }
    if (wei.device_type() != device_type) {
      return base::error::InternalError("The weight tensor " + std::to_string(i) +
                                        " has a wrong device type.");
    }
    if (wei.data_type() != data_type) {
      return base::error::InternalError("The weight tensor " + std::to_string(i) +
                                        " has a wrong data type.");
    }
  }
  return error::Success();
}

}  // namespace op