#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include "rmsnorm_kernel.cuh"
namespace kernel {
__global__ void multihead_attention_fp32() {}

static __global__ void row_rmsnorm_f32(const float* in, const float* wei, float* out,
                                       const int size, const float eps) {
  const int tid = threadIdx.x;
  const int lane_id = tid % warpSize;

  float sum = 0.0f;
  for (int i = lane_id; i < size; i += warpSize) {
    sum += in[i] * in[i];
  }

  using WarpReduce = cub::WarpReduce<float, 32>;
  __shared__ typename WarpReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = WarpReduce(temp).Reduce(sum, cub::Sum());
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  for (int i = lane_id; i < size; i += warpSize) {
    out[i] = scale * in[i] * wei[i];
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-5f;
  const int32_t size = static_cast<int32_t>(input.size());
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (size < 1024) {
    constexpr int threads_num = 128;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32<<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  } else {
    constexpr int threads_num = 1024;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32<<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  }
}
}  // namespace kernel