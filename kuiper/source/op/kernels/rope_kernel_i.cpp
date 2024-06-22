#include "rope_kernel_i.h"
#include "cpu/rope_kernel.h"
#include "cuda/rope_kernel_cu.cuh"
namespace kernel {
RoPEKernel get_rope_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rope_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rope_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}
}  // namespace kernel
