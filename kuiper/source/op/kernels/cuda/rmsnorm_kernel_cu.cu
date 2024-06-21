#include "rmsnorm_kernel_cu.cuh"
#include "utils.cuh"
#define WARP_SIZE 32
namespace kernel {

template <int NUM_THREADS>
static __global__ void row_rmsnorm_f32(const float* in, const float* wei, float* out,
                                        const int size, const float eps) {
  int tid = threadIdx.x;
  __shared__ float scale;
  float value = 0.0f;
  for (int i = tid; i < size; i += NUM_THREADS) {
    const float xi = in[i];
    value += xi * xi;
  }

  const float mean = block_reduce_sum<NUM_THREADS>(value) / static_cast<float>(size);
  if (tid == 0) scale = rsqrtf(mean + eps);
  __syncthreads();
  for (int i = tid; i < size; i += NUM_THREADS) {
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
  if (stream) {
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    if (size > 1024) {
      row_rmsnorm_f32<1024>
          <<<1, 1024, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<32><<<1, 32, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  } else {
    if (size > 1024) {
      row_rmsnorm_f32<1024><<<1, 1024>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<32><<<1, 32>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  }
}
}  // namespace kernel