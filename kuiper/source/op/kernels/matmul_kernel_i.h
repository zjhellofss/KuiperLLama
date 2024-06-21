#ifndef MATMUL_KERNEL_I_H
#define MATMUL_KERNEL_I_H
#include <cublas_v2.h>
#include <tensor/tensor.h>
namespace kernel {
struct BlasCudaConfig {
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
};

typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output,
                             const BlasCudaConfig* config);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // MATMUL_KERNEL_I_H
