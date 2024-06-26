#ifndef ADD_KERNEL_I_H
#define ADD_KERNEL_I_H
#include "cpu/add_kernel.h"
namespace kernel {
typedef void (*AddKernel)(float scale1, const tensor::Tensor& input1, float scale2,
                          const tensor::Tensor& input2, const tensor::Tensor& output,
                          void* stream);

AddKernel get_add_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // ADD_KERNEL_I_H
