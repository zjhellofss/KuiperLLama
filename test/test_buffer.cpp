#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"

TEST(test_buffer, allocate) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  { // 局部范围内，buffer退出这个花括号范围就会被释放，Buffer释放就会去调用~Buffer()
    Buffer buffer(32, alloc);
    ASSERT_NE(buffer.ptr(), nullptr);
    LOG(INFO) << "HERE1";
  } //已经要退出这个范围了
  LOG(INFO) << "HERE2";
}

TEST(test_buffer, allocate2) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  // std::shared_ptr<Buffer> buffer; //还有外部引用
  {
    std::make_shared<Buffer>(32, alloc);
  } // 退出buffer的时候，因为buffer还有外部引用
  LOG(INFO) << "HERE";
}

TEST(test_buffer, use_external) {
  using namespace base;
  auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.is_external(), true);
  delete[] ptr;
}