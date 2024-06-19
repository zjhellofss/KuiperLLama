#include "emb_kernel_i.h"
#include "cpu/emb_kernel.h"
namespace kernel {
EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU ||
      device_type == base::DeviceType::kDeviceCUDA) {
    return emb_kernel_normal;
  } else {
    LOG(FATAL) << "Unknown device type for get an embedding kernel.";
    return nullptr;
  }
}
}  // namespace kernel
