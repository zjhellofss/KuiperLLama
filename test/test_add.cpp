#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "base/buffer.h"
#include "../kuiper/source/op/kernels/add_kernel.h"

TEST(test_op, add) {
  using namespace base;
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor t2(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  tensor::Tensor out(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  for (int i = 0; i < t1.size(); ++i) {
    t1.index<float>(i) = 1;
    t2.index<float>(i) = 2;
  }

  kernel::get_add_kernel(base::DeviceType::kDeviceCPU)(t1, t2, out);
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(out.index<float>(i), 3.f);
  }
}
