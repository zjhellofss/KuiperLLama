#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/cpu/matmul_kernel.h"
#include "../source/op/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
using namespace kernel;
TEST(test_matmul_cu, matmul_linear_stream5) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor input(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, 4, 4, true, alloc_cpu);

  for (int i = 0; i < 4; ++i) {
    input.index<float>(i) = float(i);
  }

  for (int i = 0; i < 16; ++i) {
    weight.index<float>(i) = float(i);
  }
  tensor::Tensor input_cpu = input.clone();
  tensor::Tensor weight_cpu = weight.clone();

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor out_cu(base::DataType::kDataTypeFp32, 4, true, alloc_cu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, 4, true, alloc_cpu);

  CudaConfig* config = new CudaConfig;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  config->stream = stream;
  kernel::get_matmul_kernel(base::DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, config);

  kernel::get_matmul_kernel(base::DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                          config);

  out_cu.to_cpu();
  for (int i = 0; i < out_cu.size(); ++i) {
    ASSERT_EQ(out_cu.index<float>(i), out_cpu.index<float>(i));
  }
}