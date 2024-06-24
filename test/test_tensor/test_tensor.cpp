#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "../utils.cuh"
#include "base/buffer.h"

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

TEST(test_tensor, clone_cuda) {
  using namespace base;
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
  ASSERT_EQ(t1_cu.is_empty(), false);
  set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.f);

  tensor::Tensor t2_cu = t1_cu.clone();
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  cudaMemcpy(p2, t1_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  ASSERT_EQ(t2_cu.data_type(), base::DataType::kDataTypeFp32);
  ASSERT_EQ(t2_cu.size(), 32 * 32);

  t2_cu.to_cpu();
  std::memcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}

TEST(test_tensor, clone_cpu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  for (int i = 0; i < 32 * 32; ++i) {
    t1_cpu.index<float>(i) = 1.f;
  }

  tensor::Tensor t2_cpu = t1_cpu.clone();
  float* p2 = new float[32 * 32];
  std::memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
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

TEST(test_tensor, assign1) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);

  int32_t size = 32 * 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  std::shared_ptr<Buffer> buffer =
      std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr, true);
  buffer->set_device_type(DeviceType::kDeviceCPU);

  ASSERT_EQ(t1_cpu.assign(buffer), true);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
  delete[] ptr;
}