//
// Created by fss on 6/24/24.
//

#include "mha_kernel_i.h"
#include "cpu/mha_kernel.h"
namespace kernel {
MHAKernel get_mha_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU ||
      device_type == base::DeviceType::kDeviceCUDA) {
    return mha_kernel;
  } else {
    LOG(FATAL) << "Unknown device type for get an mha kernel.";
    return nullptr;
  }
}
}  // namespace kernel