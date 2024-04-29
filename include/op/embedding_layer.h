
#ifndef LC_INCLUDE_OP_EMBEDDING_LAYER_H_
#define LC_INCLUDE_OP_EMBEDDING_LAYER_H_
#include "layer.h"
namespace op {
class EmbeddingLayer : public LayerFp32Param {
 public:
  explicit EmbeddingLayer();

  base::Status check() override;

  base::Status forward() override;
};
}  // namespace op
#endif  // LC_INCLUDE_OP_EMBEDDING_LAYER_H_
