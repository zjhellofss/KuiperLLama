#ifndef KUIPER_INCLUDE_OP_H
#define KUIPER_INCLUDE_OP_H
#include "layer.h"
namespace op {
class SoftmaxLayer : public Layer {
 public:
  explicit SoftmaxLayer(base::DeviceType device_type);

  base::Status check() const override;

  base::Status base_forward() override;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_H
