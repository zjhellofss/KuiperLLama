#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"



TEST(test_buffer, allocate2) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<Buffer> buffer;
  { buffer = std::make_shared<Buffer>(32, alloc); }
  LOG(INFO) << "HERE";
  ASSERT_NE(buffer->ptr(), nullptr);
}

TEST(test_buffer, use_external) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.is_external(), true);
  delete[] ptr;
}