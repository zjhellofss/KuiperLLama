#include "scale_sum_kernel.cuh"
namespace kernel {
__global__ void scale_sum_kernel_cu_fp32(const float* value, const float* scale, float* output, 
                                         int pos, int size, int stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    float sum = 0.0f;
      for (int i = 0; i <= pos; ++i) {
        sum += value[i * stride + tid] * scale[i];
      }
    output[tid] = sum;
  }
}

void scale_sum_kernel_cu(const tensor::Tensor& value, const tensor::Tensor& scale, 
                         const tensor::Tensor& output, int pos, int size, int stride, void* stream) {
  CHECK_EQ(value.is_empty(), false);
  CHECK_EQ(scale.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  CHECK_EQ(size, value.size());
  CHECK_GE(size, scale.size());
  CHECK_EQ(size, output.size());
  int32_t thread_num = 512;
  int32_t block_num = (size + thread_num - 1) / thread_num;
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    scale_sum_kernel_cu_fp32<<<block_num, thread_num, 0, stream_>>>(
      value.ptr<float>(), scale.ptr<float>(),
      const_cast<float*>(output.ptr<float>()), pos, size, stride);
  } else {
    scale_sum_kernel_cu_fp32<<<block_num, thread_num>>>(
      value.ptr<float>(), scale.ptr<float>(),
      const_cast<float*>(output.ptr<float>()), pos, size, stride);
  }
}
}  // namespace kernel