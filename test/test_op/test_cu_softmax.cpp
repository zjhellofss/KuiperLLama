#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../kuiper/source/op/kernels/softmax_kernel_i.h"
#include "../source/op/kernels/rms_kernel_i.h"
#include "../utils.cuh"
#include "base/buffer.h"

TEST(test_softmax_cu, softmax_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  srand(0);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = rand() % 31;
  }

  tensor::Tensor in_cu = in_cpu.clone();
  in_cu.to_cuda();

  kernel::get_softmax_kernel(base::DeviceType::kDeviceCUDA)(in_cu, nullptr);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(in_cpu, nullptr);

  in_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(in_cpu.index<float>(i), in_cu.index<float>(i), 1e-5f);
  }
}


TEST(test_softmax_cu, softmax_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 72 * 151;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  srand(0);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = rand() % 31;
  }

  tensor::Tensor in_cu = in_cpu.clone();
  in_cu.to_cuda();

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCUDA)(in_cu, stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(in_cpu, nullptr);

  in_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(in_cpu.index<float>(i), in_cu.index<float>(i), 1e-5f);
  }
}


TEST(test_softmax_cu, softmax_stream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 72 * 18;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f,1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  in_cu.to_cuda();

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCUDA)(in_cu, stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(in_cpu, nullptr);
  in_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(in_cpu.index<float>(i), in_cu.index<float>(i), 1e-5f);
  }
}


TEST(test_softmax_cu, softmax_stream3) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 1;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f,1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  in_cu.to_cuda();

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCUDA)(in_cu, stream);
  kernel::get_softmax_kernel(base::DeviceType::kDeviceCPU)(in_cpu, nullptr);
  in_cu.to_cpu();

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(in_cpu.index<float>(i), in_cu.index<float>(i), 1e-5f);
  }
}