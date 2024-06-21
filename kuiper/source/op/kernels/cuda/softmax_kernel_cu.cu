#include "softmax_kernel_cu.cuh"
#include "utils.cuh"
namespace kernel {

template <const int NUM_THREADS = 128>
__global__ void row_softmax_fp32(float* in, int size) {
  int tid = threadIdx.x;
  if (tid > size) {
    return;
  }
  extern __shared__ float shared_mem[];

  // get the max value
  float max_value = -INFINITY;
  for (int i = tid; i < size; i += NUM_THREADS) {
    const float xi = in[i];
    shared_mem[i] = xi;
    max_value = max(xi, max_value);
  }
  max_value = block_reduce_max<NUM_THREADS>(max_value);

  // get the sum value
  float sum_value = 0.f;
  for (int i = tid; i < size; i += NUM_THREADS) {
    const float xi = exp(shared_mem[i] - max_value);
    shared_mem[i] = xi;
    sum_value += xi;
  }
  sum_value = block_reduce_sum<NUM_THREADS>(sum_value);

  // write back
  for (int i = tid; i < size; i += NUM_THREADS) {
    const float out = shared_mem[i] / sum_value;
    in[i] = out;
  }
}

void softmax_inplace_kernel_cu(const tensor::Tensor& input, void* stream) {
  CHECK_EQ(input.is_empty(), false);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  int32_t size = static_cast<int32_t>(input.size());
  const size_t shmem = (KUIPER_PAD(size, WARP_SIZE) + WARP_SIZE) * sizeof(float);
  if (size < 1024) {
    constexpr int threads_num = 128;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_softmax_fp32<threads_num><<<1, threads_num, shmem, stream_>>>(
          const_cast<float*>(input.ptr<float>()), size);
    } else {
      row_softmax_fp32<threads_num>
          <<<1, threads_num, shmem>>>(const_cast<float*>(input.ptr<float>()), size);
    }
  } else {
    constexpr int threads_num = 1024;
    row_softmax_fp32<threads_num>
        <<<1, threads_num, shmem>>>(const_cast<float*>(input.ptr<float>()), size);
  }
}
}  // namespace kernel