#ifndef KUIPER_INCLUDE_OP_KERNEL
#define KUIPER_INCLUDE_OP_KERNEL
#include "base/base.h"
#include "tensor/tensor.h"
namespace kernel {
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          tensor::Tensor output);

AddKernel get_add_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif