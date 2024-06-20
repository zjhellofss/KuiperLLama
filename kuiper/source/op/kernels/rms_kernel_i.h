#ifndef RMS_KERNEL_I_H
#define RMS_KERNEL_I_H
#include <tensor/tensor.h>
namespace kernel {
typedef void (*RMSNormKernel)(int32_t dim, const tensor::Tensor& input,
                              const tensor::Tensor& weight, const tensor::Tensor& output,
                              void* stream);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);
}  // namespace kernel

#endif  // RMS_KERNEL_I_H
