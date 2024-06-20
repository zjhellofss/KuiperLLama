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

template <int block_size>
static __global__ void rms_norm_f32(const float* in, const float* wei, float* out,
                                    const int dim, const float eps) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  float tmp = 0.0f;  // partial sum for thread in warp

  for (int i = tid; i < dim; i += block_size) {
    const float xi = in[row * dim + i];
    tmp += xi * xi;
  }

  // sum up partial sums
  tmp = warp_reduce_sum(tmp);
  if (block_size > WARP_SIZE) {
    __shared__ float s_sum[32];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
      s_sum[warp_id] = tmp;
    }
    __syncthreads();
    tmp = s_sum[lane_id];
    tmp = warp_reduce_sum(tmp);
  }

  const float mean = tmp / dim;
  const float scale = rsqrtf(mean + eps);

  for (int i = tid; i < dim; i += block_size) {
    out[row * dim + i] = scale * in[row * dim + i] * wei[row * dim + i];
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
  constexpr int32_t threads = 32;
  int32_t blocks = (dim + threads - 1) / threads;
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    rms_norm_f32<threads>
        <<<blocks, threads, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim, eps);
  } else {
    rms_norm_f32<threads><<<blocks, threads>>>(in_ptr, wei_ptr, out_ptr, dim, eps);
  }
}
}  // namespace kernel