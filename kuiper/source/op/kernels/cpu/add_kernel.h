#ifndef KUIPER_INCLUDE_OP_KERNEL
#define KUIPER_INCLUDE_OP_KERNEL
#include "tensor/tensor.h"
namespace kernel {
void add_kernel_cpu(float scale1, const tensor::Tensor& input1, float scale2,
                    const tensor::Tensor& input2, const tensor::Tensor& output,
                    void* stream = nullptr);
}  // namespace kernel
#endif