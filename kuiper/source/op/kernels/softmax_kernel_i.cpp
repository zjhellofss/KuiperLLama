#include "softmax_kernel_i.h"
#include "cpu/softmax_kernel.h"
#include "cuda/softmax_kernel_cu.cuh"
namespace kernel {
SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return softmax_inplace_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return softmax_inplace_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an softmax kernel.";
    return nullptr;
  }
}

}  // namespace kernel