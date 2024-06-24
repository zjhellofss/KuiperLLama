#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
namespace kernel {
struct BlasCudaConfig {
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
