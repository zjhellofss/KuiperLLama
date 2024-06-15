#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"

TEST(test_buffer, allocate) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  Buffer buffer(32, alloc);
  CHECK_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, use_external) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  CHECK_EQ(buffer.is_external(), true);
  delete[] ptr;
}