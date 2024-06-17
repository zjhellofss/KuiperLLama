#include <glog/logging.h>
#include "test_cu.cuh"
__global__ void test_function_cu(float* cu_arr, int32_t size) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) {
    return;
  }
  cu_arr[tid] = 1.f;
}

void test_function(float* arr, int32_t size) {
  if (!arr) {
    return;
  }
  float* cu_arr = nullptr;
  cudaMalloc(&cu_arr, sizeof(float) * size);

  test_function_cu<<<1, size>>>(cu_arr, size);
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  CHECK_EQ(err, cudaSuccess);

  cudaMemcpy(arr, cu_arr, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(cu_arr);
}
