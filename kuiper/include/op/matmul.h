//
// Created by hello on 2024/5/2.
//

#ifndef KUIPER_INCLUDE_OP_MATMUL_H_
#define KUIPER_INCLUDE_OP_MATMUL_H_
#include <base/cuda_config.h>
#include "layer.h"
namespace op {
class MatmulLayer : public LayerFp32Param {
 public:
  explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1);

  base::Status check() const override;

  base::Status forward() override;

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_MATMUL_H_
