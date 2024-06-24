#include "scale_tensor_i.h"
#include "./cpu/scale_kernel.h"
#include "./cuda/scale_kernel_cu.cuh"
namespace kernel {

ScaleKernel get_scale_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return scale_inplace_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return scale_inplace_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}
}  // namespace kernel
