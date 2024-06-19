#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
#include "test_cu.cuh"

TEST(test_buffer, allocate) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  Buffer buffer(32, alloc);
  ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, use_external) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.is_external(), true);
  delete[] ptr;
}

TEST(test_buffer, cuda_memcpy1) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  Buffer buffer(size * sizeof(float), nullptr, ptr, true);
  buffer.set_device_type(DeviceType::kDeviceCPU);
  ASSERT_EQ(buffer.is_external(), true);

  Buffer cu_buffer(size * sizeof(float), alloc_cu);
  cu_buffer.copy_from(buffer);

  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, cu_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    // ptr[i] = float(i);
    ASSERT_EQ(ptr2[i], float(i));
  }

  delete[] ptr;
  delete[] ptr2;
}

TEST(test_buffer, cuda_memcpy2) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  Buffer buffer(size * sizeof(float), nullptr, ptr, true);
  buffer.set_device_type(DeviceType::kDeviceCPU);
  ASSERT_EQ(buffer.is_external(), true);

  // cpu to cuda
  Buffer cu_buffer(size * sizeof(float), alloc_cu);
  cu_buffer.copy_from(buffer);

  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, cu_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], float(i));
  }

  delete[] ptr;
  delete[] ptr2;
}

TEST(test_buffer, cuda_memcpy3) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32;
  Buffer cu_buffer1(size * sizeof(float), alloc_cu);
  Buffer cu_buffer2(size * sizeof(float), alloc_cu);

  set_value_cu((float*)cu_buffer2.ptr(), size);
  // cu to cu
  ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCUDA);

  cu_buffer1.copy_from(cu_buffer2);

  float* ptr2 = new float[size];
  cudaMemcpy(ptr2, cu_buffer1.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], 1.f);
  }
  delete[] ptr2;
}

TEST(test_buffer, cuda_memcpy4) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  int32_t size = 32;
  Buffer cu_buffer1(size * sizeof(float), alloc_cu);
  Buffer cu_buffer2(size * sizeof(float), alloc);
  ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
  ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCPU);

  // cu to cpu
  set_value_cu((float*)cu_buffer1.ptr(), size);
  cu_buffer2.copy_from(cu_buffer1);

  float* ptr2 = (float*)cu_buffer2.ptr();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(ptr2[i], 1.f);
  }
}