#ifndef SCALE_TENSOR_I_H
#define SCALE_TENSOR_I_H
#include <tensor/tensor.h>
namespace kernel {
typedef void (*ScaleKernel)(float scale, const tensor::Tensor& input, void* stream);

ScaleKernel get_scale_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // SCALE_TENSOR_I_H
