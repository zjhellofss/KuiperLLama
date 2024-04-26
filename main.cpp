#include <glog/logging.h>

#include <iostream>
#include <memory>

#include "base/alloc.h"
#include "base/buffer.h"
#include "tensor/tensor.h"

int main() {
  std::shared_ptr<CPUDeviceAllocator> alloc = std::make_shared<CPUDeviceAllocator>();
  Tensor<float> tensor(1, 2, 3, 4);
  tensor.allocate(alloc);
  tensor.allocate(alloc, true);
  tensor.allocate(alloc, false);

  tensor.reset_dims({4, 5, 6});
  tensor.allocate(alloc, false);

  std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(481, alloc);
  tensor.assign(buffer);
  return 0;
}