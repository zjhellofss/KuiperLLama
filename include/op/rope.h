#ifndef KUIPER_INCLUDE_OP_ROPE_H_
#define KUIPER_INCLUDE_OP_ROPE_H_
#include "layer.h"
namespace op {
class RoPELayer : public Layer {
 public:
  explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size);

  base::Status check() const override;

  base::Status base_forward() override;

 private:
  int32_t dim_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_size_ = 0;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_ROPE_H_
