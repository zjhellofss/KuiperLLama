#include <glog/logging.h>
#include <iostream>
#include <memory>
#include "base/alloc.h"
#include "base/buffer.h"
#include "tensor/tensor.h"

int main() {
  std::shared_ptr<CPUDeviceAllocator> alloc = std::make_shared<CPUDeviceAllocator>();
  Tensor tensor(DataType::kDataTypeFp32, 1, 2, 3, 4);
  tensor.allocate(alloc);
  tensor.allocate(alloc);

  tensor.reset(DataType::kDataTypeFp32, {4, 5, 6});
  tensor.allocate(alloc);

  tensor.reshape({11, 12, 13});
  const auto& strides = tensor.strides();
  return 0;
}