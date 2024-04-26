#include <glog/logging.h>

#include <iostream>
#include <memory>

#include "base/alloc.h"
#include "base/buffer.h"

int main() {
  std::shared_ptr<DeviceAllocator> alloc =
      std::make_shared<CPUDeviceAllocator>();
  float *a = static_cast<float *>(malloc(3 * sizeof(float)));
  std::shared_ptr<Buffer> buffer =
      std::make_shared<Buffer>(3 * sizeof(float), alloc, a, false);
  free(a);
  return 0;
}