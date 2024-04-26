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
  explicit Layer(LayerType layer_type, std::string layer_name = "");

  virtual LayerStatus Init() = 0;

  virtual LayerStatus Forward() = 0;

  virtual void set_input(int32_t idx, const Tensor<float> &input) = 0;

  virtual void set_output(int32_t idx, const Tensor<float> &output) = 0;

  virtual Tensor<float> get_input(int32_t idx) const = 0;

  virtual Tensor<float> get_output(int32_t idx) const = 0;

  virtual void set_weight(int32_t idx, const Tensor<float> &weight) = 0;

  virtual Tensor<float> get_weight(int32_t idx) const = 0;
 private:
  LayerType layer_type_;
  std::string layer_name_;
};

class LayerFp32 : public Layer {
 public:
  LayerStatus Init() override;

  LayerStatus Forward() override;

  void set_input(int32_t idx, const Tensor<float> &input) override;

  void set_output(int32_t idx, const Tensor<float> &output) override;

  Tensor<float> get_input(int32_t idx) const override;

  Tensor<float> get_output(int32_t idx) const override;

  void set_weight(int32_t idx, const Tensor<float> &weight) override;

  Tensor<float> get_weight(int32_t idx) const override;
 private:
  std::vector<Tensor<float>> weights_;
  std::vector<Tensor<float>> inputs_;
  std::vector<Tensor<float>> outputs_;
};
#endif //LC_INCLUDE_OP_LAYER_H_
