//
// Created by fss on 9/19/24.
//
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "../utils.cuh"
#include "base/buffer.h"
TEST(test_buffer, use_external1) {
  using namespace base;
  auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  CHECK_EQ(buffer.is_external(), true);
  cudaFree(buffer.ptr());
}