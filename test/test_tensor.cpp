#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "base/buffer.h"
#include "test_cu.cuh"

TEST(test_tensor, to_cpu) {
  using namespace base;
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
  ASSERT_EQ(t1_cu.is_empty(), false);
  set_value_cu(t1_cu.ptr<float>(), 32 * 32);

  t1_cu.to_cpu();
  ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCPU);
  float* cpu_ptr = t1_cu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(cpu_ptr + i), 1.f);
  }
}

TEST(test_tensor, to_cu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  float* p1 = t1_cpu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    *(p1 + i) = 1.f;
  }

  t1_cpu.to_cuda();
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(p2 + i), 1.f);
  }
  delete[] p2;
}