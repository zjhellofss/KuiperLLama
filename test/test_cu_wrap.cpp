#include <glog/logging.h>
#include <gtest/gtest.h>
#include <armadillo>
#include "test_cu.cuh"

TEST(test_cu, test_function) {
  int32_t size = 32;
  float* ptr = new float[size];
  test_function(ptr, size);
  for (int32_t i = 0; i < size; ++i) {
    ASSERT_EQ(ptr[i], 1.f);
  }
  delete[] ptr;
}
