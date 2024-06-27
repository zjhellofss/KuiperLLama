#include "scale_kernel.cuh"
namespace kernel {
static __global__ void scale_inplace_fp32(float scale, float* ptr, int32_t size) {
  int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx > size) {
    return;
  }
  extern __shared__ float shmem[];
  shmem[threadIdx.x] = ptr[idx];
  __syncthreads();
  ptr[idx] = shmem[threadIdx.x] * scale;
}

void scale_inplace_cu(float scale, const tensor::Tensor& tensor, void* stream) {
  CHECK(tensor.is_empty() == false);
  CHECK(tensor.device_type() == base::DeviceType::kDeviceCUDA);
  int32_t size = static_cast<int32_t>(tensor.size());
  int32_t threads = 128;
  int32_t blocks = (size + threads - 1) / threads;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    scale_inplace_fp32<<<blocks, threads, sizeof(float) * size, stream_>>>(
        scale, const_cast<float*>(tensor.ptr<float>()), size);
  } else {
    scale_inplace_fp32<<<blocks, threads, sizeof(float) * size>>>(
        scale, const_cast<float*>(tensor.ptr<float>()), size);
  }
}
}  // namespace kernel
