#ifndef SOFTMAX_KERNEL_CU_CUH
#define SOFTMAX_KERNEL_CU_CUH
#include <tensor/tensor.h>
namespace kernel {
void softmax_inplace_kernel_cu(const tensor::Tensor& input, void* stream);
}
#endif  // SOFTMAX_KERNEL_CU_CUH
