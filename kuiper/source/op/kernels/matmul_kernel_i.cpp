#include "matmul_kernel_i.h"
#include "cpu/matmul_kernel.h"
#include "cuda/matmul_kernel_cu.cuh"
namespace kernel {
MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return matmul_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return matmul_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an matmul kernel.";
    return nullptr;
  }
}
}  // namespace kernel