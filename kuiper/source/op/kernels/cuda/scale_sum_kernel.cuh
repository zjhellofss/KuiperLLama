#ifndef SCALE_SUM_KERNEL_CU_H
#define SCALE_SUM_KERNEL_CU_H
#include <tensor/tensor.h>
namespace kernel {
void scale_sum_kernel_cu(const tensor::Tensor& value, const tensor::Tensor& scale, 
                         const tensor::Tensor& output, int t, int d, int stride, 
                         void* stream = nullptr);
}
#endif //SCALE_REDUCE_KERNEL_CU_H
