
#ifndef LC_INCLUDE_OP_EMBEDDING_LAYER_H_
#define LC_INCLUDE_OP_EMBEDDING_LAYER_H_
#include "layer.h"
namespace op {
class EmbeddingLayer : public LayerFp32Param {
 public:
  explicit EmbeddingLayer(int32_t dim, int32_t seq_len, int32_t vocab_size);

  base::Status check() override;

  base::Status base_forward() override;

 private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
};
}  // namespace op
#endif  // LC_INCLUDE_OP_EMBEDDING_LAYER_H_
