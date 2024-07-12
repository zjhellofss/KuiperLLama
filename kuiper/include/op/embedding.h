
#ifndef KUIPER_INCLUDE_OP_EMBEDDING_H_
#define KUIPER_INCLUDE_OP_EMBEDDING_H_
#include "layer.h"
namespace op {
struct EmbeddingOutput {
  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
};

class EmbeddingLayer : public LayerParam {
 public:
  explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                          int32_t vocab_size);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_EMBEDDING_H_
