#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/matmul_kernel_i.h"
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../utils.cuh"
#include "base/buffer.h"
using namespace kernel;
TEST(test_matmul_cu, matmul_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32, 2, 3, true, alloc_cpu);
  tensor::Tensor t2_cpu(base::DataType::kDataTypeFp32, 3, 4, true, alloc_cpu);
  // t1_cpu
  // 1 2 3
  // 4 5 6
  for (int i = 1; i <= 6; ++i) {
    t1_cpu.index<float>(i - 1) = float(i);
  }

  // t2 cpu
  // 3  4  5  6
  // 7  8  9  10
  // 11 12 13 14

  for (int i = 3; i <= 14; ++i) {
    t2_cpu.index<float>(i - 3) = float(i);
  }
  t1_cpu.to_cuda();
  t2_cpu.to_cuda();

  tensor::Tensor out(base::DataType::kDataTypeFp32, 2, 4, true, alloc_cu);
  cublasHandle_t handle;
  cublasCreate(&handle);

  kernel::BlasCudaConfig* config = new kernel::BlasCudaConfig;
  config->handle = handle;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(t2_cpu, t1_cpu, out, 2.f,
                                                           config);
  cudaDeviceSynchronize();
  float* output = new float[8];
  cudaMemcpy(output, out.ptr<float>(), 8 * sizeof(float), cudaMemcpyDeviceToHost);
  ASSERT_EQ(output[0], 50 * 2.f);
  ASSERT_EQ(output[1], 56 * 2.f);
  ASSERT_EQ(output[2], 62 * 2.f);
  ASSERT_EQ(output[3], 68 * 2.f);

  ASSERT_EQ(output[4], 113 * 2.f);
  ASSERT_EQ(output[5], 128 * 2.f);
  ASSERT_EQ(output[6], 143 * 2.f);
  ASSERT_EQ(output[7], 158 * 2.f);
  delete[] output;
}

TEST(test_matmul_cu, matmul_linear_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32, 2, 3, true, alloc_cpu);
  tensor::Tensor t2_cpu(base::DataType::kDataTypeFp32, 3, 1, true, alloc_cpu);
  // t1_cpu
  // 1 2 3
  // 4 5 6
  for (int i = 1; i <= 6; ++i) {
    t1_cpu.index<float>(i - 1) = float(i);
  }

  // t2 cpu
  // 3  4  5
  for (int i = 3; i <= 5; ++i) {
    t2_cpu.index<float>(i - 3) = float(i);
  }
  t1_cpu.to_cuda();
  t2_cpu.to_cuda();

  tensor::Tensor out(base::DataType::kDataTypeFp32, 2, 1, true, alloc_cu);
  cublasHandle_t handle;
  cublasCreate(&handle);

  BlasCudaConfig* config = new BlasCudaConfig;
  config->handle = handle;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(t2_cpu, t1_cpu, out, 3.f,
                                                           config);
  cudaDeviceSynchronize();
  float* output = new float[2];
  cudaMemcpy(output, out.ptr<float>(), 2 * sizeof(float), cudaMemcpyDeviceToHost);

  ASSERT_EQ(output[0], 26 * 3.f);
  ASSERT_EQ(output[1], 62 * 3.f);

  delete[] output;
}

TEST(test_matmul_cu, matmul_linear_nostream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32, 2, 3, true, alloc_cpu);
  tensor::Tensor t2_cpu(base::DataType::kDataTypeFp32, 3, 1, true, alloc_cpu);
  // t1_cpu
  // 1 2 3
  // 4 5 6
  for (int i = 1; i <= 6; ++i) {
    t1_cpu.index<float>(i - 1) = float(i);
  }

  // t2 cpu
  // 3  4  5
  for (int i = 3; i <= 5; ++i) {
    t2_cpu.index<float>(i - 3) = float(i);
  }
  t1_cpu.to_cuda();
  t2_cpu.to_cuda();

  tensor::Tensor out(base::DataType::kDataTypeFp32, 2, true, alloc_cu);
  cublasHandle_t handle;
  cublasCreate(&handle);

  BlasCudaConfig* config = new BlasCudaConfig;
  config->handle = handle;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(t2_cpu, t1_cpu, out, 1.f,
                                                           config);
  cudaDeviceSynchronize();
  float* output = new float[2];
  cudaMemcpy(output, out.ptr<float>(), 2 * sizeof(float), cudaMemcpyDeviceToHost);

  ASSERT_EQ(output[0], 26);
  ASSERT_EQ(output[1], 62);

  delete[] output;
}

TEST(test_matmul_cu, matmul_linear_nostream3) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32, 2, 3, true, alloc_cpu);
  tensor::Tensor t2_cpu(base::DataType::kDataTypeFp32, 3, true, alloc_cpu);
  // t1_cpu
  // 1 2 3
  // 4 5 6
  for (int i = 1; i <= 6; ++i) {
    t1_cpu.index<float>(i - 1) = float(i);
  }

  // t2 cpu
  // 3  4  5
  for (int i = 3; i <= 5; ++i) {
    t2_cpu.index<float>(i - 3) = float(i);
  }
  t1_cpu.to_cuda();
  t2_cpu.to_cuda();

  tensor::Tensor out(base::DataType::kDataTypeFp32, 2, 1, true, alloc_cu);
  cublasHandle_t handle;
  cublasCreate(&handle);

  BlasCudaConfig* config = new BlasCudaConfig;
  config->handle = handle;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(t2_cpu, t1_cpu, out, 1.f,
                                                           config);
  cudaDeviceSynchronize();
  float* output = new float[2];
  cudaMemcpy(output, out.ptr<float>(), 2 * sizeof(float), cudaMemcpyDeviceToHost);

  ASSERT_EQ(output[0], 26);
  ASSERT_EQ(output[1], 62);

  delete[] output;
}

TEST(test_matmul_cu, matmul_linear_stream1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32, 2, 4, true, alloc_cpu);
  tensor::Tensor t2_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  // t1_cpu
  // 1 2 3 4
  // 5 6 7 8
  for (int i = 1; i <= 8; ++i) {
    t1_cpu.index<float>(i - 1) = float(i);
  }

  // t2 cpu
  // 3  4  5   6
  for (int i = 3; i <= 6; ++i) {
    t2_cpu.index<float>(i - 3) = float(i);
  }
  t1_cpu.to_cuda();
  t2_cpu.to_cuda();

  tensor::Tensor out(base::DataType::kDataTypeFp32, 2, 1, true, alloc_cu);
  cublasHandle_t handle;
  cublasCreate(&handle);

  BlasCudaConfig* config = new BlasCudaConfig;
  config->handle = handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(t2_cpu, t1_cpu, out, 2.f,
                                                           config);
  cudaDeviceSynchronize();
  float* output = new float[2];
  cudaMemcpy(output, out.ptr<float>(), 2 * sizeof(float), cudaMemcpyDeviceToHost);

  ASSERT_EQ(output[0], 50 * 2);
  ASSERT_EQ(output[1], 122 * 2);

  delete[] output;
}

TEST(test_matmul_cu, matmul_linear_stream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor t1_cpu(base::DataType::kDataTypeFp32, 2, 3, true, alloc_cpu);
  tensor::Tensor t2_cpu(base::DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);
  // t1_cpu
  // 1 2 3
  // 4 5 6
  for (int i = 1; i <= 6; ++i) {
    t1_cpu.index<float>(i - 1) = float(i);
  }

  // t2 cpu
  // 3 4  5
  // 6 7  8
  // 9 10 11
  for (int i = 3; i <= 11; ++i) {
    t2_cpu.index<float>(i - 3) = float(i);
  }
  t1_cpu.to_cuda();
  t2_cpu.to_cuda();

  tensor::Tensor out(base::DataType::kDataTypeFp32, 2, 3, true, alloc_cu);
  cublasHandle_t handle;
  cublasCreate(&handle);

  BlasCudaConfig* config = new BlasCudaConfig;
  config->handle = handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(t2_cpu, t1_cpu, out, 2.f,
                                                           config);
  cudaDeviceSynchronize();
  float* output = new float[2];
  cudaMemcpy(output, out.ptr<float>(), 6 * sizeof(float), cudaMemcpyDeviceToHost);

  ASSERT_EQ(output[0], 42 * 2);
  ASSERT_EQ(output[1], 48 * 2);
  ASSERT_EQ(output[2], 54 * 2);
  ASSERT_EQ(output[3], 96 * 2);
  ASSERT_EQ(output[4], 111 * 2);
  ASSERT_EQ(output[5], 126 * 2);
  delete[] output;
}