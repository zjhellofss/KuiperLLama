#include <gtest/gtest.h>
#include "tensor/tensor.h"

TEST(test_tensor, init1) {
  using namespace base;
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  ASSERT_EQ(t1.is_empty(), false);
}

TEST(test_tensor, init2) {
  using namespace base;
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, false, alloc_cu);
  ASSERT_EQ(t1.is_empty(), true);
}

TEST(test_tensor, init3) {
  using namespace base;
  float* ptr = new float[32];
  ptr[0] = 31;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.ptr<float>(), ptr);
  ASSERT_EQ(*t1.ptr<float>(), 31);
  delete[] ptr;
}

TEST(test_tensor, init4) {
  using namespace base;
  float* ptr = new float[32];
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();
  ptr[0] = 31;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, alloc_cu, ptr);
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.ptr<float>(), ptr);
  ASSERT_EQ(*t1.ptr<float>(), 31);
  delete[] ptr;
}

TEST(test_tensor, index) {
  using namespace base;
  float* ptr = new float[32];
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();
  ptr[0] = 31;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);
  void* p1 = t1.ptr<void>();
  p1 += 1;

  float* f1 = t1.ptr<float>();
  f1 += 1;
  ASSERT_NE(f1, p1);
  delete[] ptr;
}

TEST(test_tensor, dims_stride) {
  using namespace base;
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, 32, 3, true, alloc_cu);
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.get_dim(0), 32);
  ASSERT_EQ(t1.get_dim(1), 32);
  ASSERT_EQ(t1.get_dim(2), 3);

  const auto& strides = t1.strides();
  ASSERT_EQ(strides.at(0), 32 * 3);
  ASSERT_EQ(strides.at(1), 3);
  ASSERT_EQ(strides.at(2), 1);
}
