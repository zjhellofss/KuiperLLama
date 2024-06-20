#include "rmsnorm_kernel_cu.cuh"
#define WARP_SIZE 32
namespace kernel {
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
  }
  return x;
}

static __global__ void rms_norm_f32(const float* in, const float* wei, float* out,
                                    const int dim, const float eps) {
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  float value = 0.0f;  // partial sum for thread in warp
  for (int i = tid; i < dim; i += block_size) {
    const float xi = in[i];
    value += xi * xi;
  }

  // sum up partial sums
  value = warp_reduce_sum(value);
  if (block_size > WARP_SIZE) {
    __shared__ float shared_arr[WARP_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
      shared_arr[warp_id] = value;
    }
    __syncthreads();
    value = shared_arr[lane_id];
    value = warp_reduce_sum(value);
  }

  const float mean = value / static_cast<float>(dim);
  const float scale = rsqrtf(mean + eps);

  for (int i = tid; i < dim; i += block_size) {
    out[i] = scale * in[i] * wei[i];
  }
}

void rmsnorm_kernel_cu(int32_t dim, const tensor::Tensor& input,
                       const tensor::Tensor& weight, const tensor::Tensor& output,
                       void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-5f;
  const int32_t blocks = 1;
  int32_t threads = 32;
  if (dim > 1024) {
    threads = 1024;
  }
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    rms_norm_f32<<<blocks, threads, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim, eps);
  } else {
    rms_norm_f32<<<blocks, threads>>>(in_ptr, wei_ptr, out_ptr, dim, eps);
  }
}
}  // namespace kernel