#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/add_kernel_i.h"
#include "../utils.cuh"
#include "base/buffer.h"
TEST(test_add_cu, add1_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }

  delete[] output;
}

TEST(test_add_cu, add1_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.f);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, stream);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(output[i], 5.f);
  }
  cudaStreamDestroy(stream);
  delete[] output;
}

TEST(test_add_cu, add_align1) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 13;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);

  set_value_cu(static_cast<float*>(t1.get_buffer()->ptr()), size, 2.1f);
  set_value_cu(static_cast<float*>(t2.get_buffer()->ptr()), size, 3.3f);

  kernel::get_add_kernel(base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();
  float* output = new float[size];
  cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(output[i], 5.4f,0.1f);
  }

  delete[] output;
}