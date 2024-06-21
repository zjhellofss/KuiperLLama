#ifndef MATMUL_KERNEL_CU_CUH
#define MATMUL_KERNEL_CU_CUH
#include "../matmul_kernel_i.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output,
                      const BlasCudaConfig* config = nullptr);
}

#endif  // MATMUL_KERNEL_CU_CUH
