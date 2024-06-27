#include "rmsnorm_kernel.cuh"
#include "utils.cuh"
#define WARP_SIZE 32
namespace kernel {

static __global__ void row_rmsnorm_f32(const float* in, const float* wei, float* out,
                                       const int size, const float eps) {
  const int tid = threadIdx.x;
  const int lane_id = tid % warpSize;

  float sum = 0.0f;
  for (int i = lane_id; i < size; i += warpSize) {
    sum += in[i] * in[i];
  }

  sum = warp_reduce_sum(sum);
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
      row_rmsnorm_f32<<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size,
                                                      eps);
    } else {
      row_rmsnorm_f32<<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  } else {
    constexpr int threads_num = 1024;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32<<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size,
                                                      eps);
    } else {
      row_rmsnorm_f32<<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  }
}
}  // namespace kernel