#ifndef KUIPER_INLCUDE_MHA_H
#define KUIPER_INLCUDE_MHA_H
#include "layer.h"
#include "softmax.h"
namespace op {
class MultiHeadAttention : public Layer {
 public:
  explicit MultiHeadAttention(base::DeviceType device_type, int32_t kv_mul, int32_t kv_dim,
                              int32_t seq_len, int32_t head_num, int32_t head_size);

  base::Status check() const override;

  void set_pos(int32_t pos);

  void set_layer_index(int32_t layer_index);

  base::Status base_forward() override;

 private:
  int32_t layer_index_ = 0;
  int32_t pos_ = 0;
  int32_t kv_mul_ = 0;
  int32_t kv_dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t head_num_ = 0;
  int32_t head_size_ = 0;
  std::unique_ptr<op::SoftmaxLayer> softmax_;
};
}  // namespace op
#endif  // KUIPER_INLCUDE_MHA_H
