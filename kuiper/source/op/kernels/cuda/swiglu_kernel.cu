#include <tensor/tensor.h>
#include "swiglu_kernel.cuh"
namespace kernel {
__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
  int tid = threadIdx.x;
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx > size) {
    return;
  }
  extern __shared__ float smem1[];
  extern __shared__ float smem2[];

  smem1[tid] = in1[idx];
  smem2[tid + size] = in2[idx];
  __syncthreads();

  float value = 1.0f / (1.0f + exp(-smem1[tid]));
  smem1[tid] = smem1[tid] * value;

  out[idx] = smem1[tid] * smem2[size + tid];
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      const tensor::Tensor& output, void* stream) {
  CHECK_EQ(input1.is_empty(), false);
  CHECK(input1.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(input2.is_empty(), false);
  CHECK(input2.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK_EQ(output.is_empty(), false);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

  int size = static_cast<int32_t>(input1.size());
  int threads = 128;
  int blocks = (size + threads - 1) / threads;
  const size_t shmem = size * sizeof(float) * 2;
  if (!stream) {
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  } else {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<blocks, threads, shmem, stream_>>>(
        size, input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()));
  }
}
}  // namespace kernel