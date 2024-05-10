#ifndef LC_INCLUDE_OP_ADD_H
#define LC_INCLUDE_OP_ADD_H
#include "layer.h"
namespace op {
class VecAddLayer : public op::Layer {
 public:
  explicit VecAddLayer();

  base::Status check() const override;

  base::Status base_forward() override;
};
}  // namespace op
#endif  // LC_INCLUDE_OP_ADD_H