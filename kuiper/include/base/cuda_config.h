#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  ~CudaConfig() {
    if (handle) {
      cublasDestroy(handle);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
  }
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
