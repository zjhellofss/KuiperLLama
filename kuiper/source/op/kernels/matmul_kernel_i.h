#ifndef MATMUL_KERNEL_I_H
#define MATMUL_KERNEL_I_H
#include <tensor/tensor.h>
#include "cpu/matmul_kernel.h"
namespace kernel {
typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale,
                             const CudaConfig* config);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // MATMUL_KERNEL_I_H
