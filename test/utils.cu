#include <glog/logging.h>
#include "utils.cuh"
__global__ void test_function_cu(float* cu_arr, int32_t size, float value) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= size) {
    return;
  }
  cu_arr[tid] = value;
}

void test_function(float* arr, int32_t size, float value) {
  if (!arr) {
    return;
  }
  float* cu_arr = nullptr;
  cudaMalloc(&cu_arr, sizeof(float) * size);
  cudaDeviceSynchronize();
  const cudaError_t err2 = cudaGetLastError();
  test_function_cu<<<1, size>>>(cu_arr, size, value);
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  CHECK_EQ(err, cudaSuccess);

  cudaMemcpy(arr, cu_arr, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(cu_arr);
}

void set_value_cu(float* arr_cu, int32_t size, float value) {
  int32_t threads_num = 512;
  int32_t block_num = (size + threads_num - 1) / threads_num;
  cudaDeviceSynchronize();
  const cudaError_t err2 = cudaGetLastError();
  test_function_cu<<<block_num, threads_num>>>(arr_cu, size, value);
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  CHECK_EQ(err, cudaSuccess);
}
