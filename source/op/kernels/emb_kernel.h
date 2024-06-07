//
// Created by fss on 24-6-7.
//

#ifndef KUIPER_INFER_EMB_KERNEL_H
#define KUIPER_INFER_EMB_KERNEL_H
#include "base/base.h"
#include "tensor/tensor.h"
namespace kernel {
typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                tensor::Tensor output, int32_t vocab_size);

EmbeddingKernel get_emb_kernel(base::DeviceType device_type);
}  // namespace kernel
#endif  // KUIPER_INFER_EMB_KERNEL_H
