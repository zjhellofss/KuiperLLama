#ifndef ADD_CU_H
#define ADD_CU_H
#include "tensor/tensor.h"
namespace kernel {
void add_kernel_cu(float scale1, const tensor::Tensor& input1, float scale2,
                   const tensor::Tensor& input2, const tensor::Tensor& output,
                   void* stream = nullptr);
}  // namespace kernel
#endif  // ADD_CU_H
