#ifndef LC_INCLUDE_OP_H
#define LC_INCLUDE_OP_H
#include "layer.h"
namespace op {
class SoftmaxLayer : public Layer {
 public:
  explicit SoftmaxLayer();

  base::Status check() const override;

  base::Status base_forward() override;
};
}  // namespace op
#endif  // LC_INCLUDE_OP_H
