#ifndef LC_INCLUDE_OP_RMSNORM_H_
#define LC_INCLUDE_OP_RMSNORM_H_
#include "layer.h"
namespace op {
class RmsNormLayer : public LayerFp32Param {
 public:
  explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

  base::Status check() const override;

  base::Status base_forward() override;

 private:
  int32_t dim_ = 0;
};
}  // namespace op
#endif  // LC_INCLUDE_OP_RMSNORM_H_
