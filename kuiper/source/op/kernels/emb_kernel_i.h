#ifndef EMB_KERNEL_I_H
#define EMB_KERNEL_I_H
#include <tensor/tensor.h>
namespace kernel {
typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size,
                                void* stream);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // EMB_KERNEL_I_H
