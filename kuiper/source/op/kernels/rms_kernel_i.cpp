#include "rms_kernel_i.h"
#include "cpu/rmsnorm_kernel.h"
#include "cuda/rmsnorm_kernel_cu.cuh"
kernel::RMSNormKernel kernel::get_rmsnorm_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an rmsnorm kernel.";
    return nullptr;
  }
}
