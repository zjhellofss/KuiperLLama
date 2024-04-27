#ifndef LC_INCLUDE_OP_LAYER_H_
#define LC_INCLUDE_OP_LAYER_H_
#include <utility>
#include <vector>
#include "tensor/tensor.h"

enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
};

enum class LayerStatus : uint8_t {
  kForwardSuccess = 0,
  kFunctionUnImplement = 1,
};

class Layer {
 public:
  explicit Layer(LayerType layer_type, DataType data_type, std::string layer_name = "");

  DataType data_type() const;

  LayerType layer_type() const;

  virtual LayerStatus Init() = 0;

  virtual LayerStatus Forward() = 0;

  virtual void set_input(int32_t idx, const Tensor& input) = 0;

  virtual void set_output(int32_t idx, const Tensor& output) = 0;

  virtual Tensor get_input(int32_t idx) const = 0;

  virtual Tensor get_output(int32_t idx) const = 0;

  virtual void set_weight(int32_t idx, const Tensor& weight) = 0;

  virtual Tensor get_weight(int32_t idx) const = 0;

  virtual void reset_input_size(size_t size) = 0;

  virtual void reset_output_size(size_t size) = 0;

 private:
  std::string layer_name_;
  DataType data_type_ = DataType::kDataTypeUnknown;
  LayerType layer_type_ = LayerType::kLayerUnknown;
};

class ParamLayerFp32 : public Layer {
 public:
  explicit ParamLayerFp32(LayerType layer_type, std::string layer_name = "");

  LayerStatus Init() override;

  LayerStatus Forward() override;

  void set_input(int32_t idx, const Tensor& input) override;

  void set_output(int32_t idx, const Tensor& output) override;

  Tensor get_input(int32_t idx) const override;

  Tensor get_output(int32_t idx) const override;

  void set_weight(int32_t idx, const Tensor& weight) override;

  Tensor get_weight(int32_t idx) const override;

  void set_weight(int32_t idx, const std::vector<int32_t>& dims, const float* weight_ptr);

  void reset_weight_size(size_t size);

  void reset_input_size(size_t size) override;

  void reset_output_size(size_t size) override;

 private:
  std::vector<Tensor> weights_;
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};
#endif  // LC_INCLUDE_OP_LAYER_H_
