#ifndef MATMUL_CUTLASS_CUH
#define MATMUL_CUTLASS_CUH

#ifdef USE_CUTLASS

#include "../kernels_interface.h"
#include "tensor/tensor.h"
namespace kernel {
void matmul_cutlass(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, float scale = 1.f,
                    const CudaConfig* config = nullptr);
}  // namespace kernel

#endif  // USE_CUTLASS

#endif  // MATMUL_CUTLASS_CUH
