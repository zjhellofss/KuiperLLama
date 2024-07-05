#include <cub/block/block_reduce.cuh>
#include "softmax_kernel.cuh"
namespace kernel {
__global__ void row_softmax_fp32(const float* in, int size) {
  const int tid = threadIdx.x;
  const int lane_id = tid % warpSize;
  const float* x = in;
  float* y = const_cast<float*>(in);

  float max_val = -INFINITY;
  for (int i = lane_id; i < size; i += warpSize) {
    max_val = fmaxf(max_val, x[i]);
  }

  using WarpReduce = cub::WarpReduce<float>;
  __shared__ WarpReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = WarpReduce(temp).Reduce(max_val, cub::Max());

  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = lane_id; i < size; i += warpSize) {
    sum += expf(x[i] - max_val);
  }

  sum = WarpReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  for (int i = lane_id; i < size; i += warpSize) {
    y[i] = expf(x[i] - max_val) / sum;
  }
}

void softmax_inplace_kernel_cu(const tensor::Tensor& input, void* stream) {
  CHECK_EQ(input.is_empty(), false);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
  int32_t size = static_cast<int32_t>(input.size());
  if (size < 1024) {
    constexpr int threads_num = 128;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_softmax_fp32<<<1, threads_num, 0, stream_>>>(input.ptr<float>(), size);
    } else {
      row_softmax_fp32<<<1, threads_num>>>(const_cast<float*>(input.ptr<float>()), size);
    }
  } else {
    constexpr int threads_num = 1024;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_softmax_fp32<<<1, threads_num, 0, stream_>>>(input.ptr<float>(), size);
    } else {
      row_softmax_fp32<<<1, threads_num>>>(input.ptr<float>(), size);
    }
  }
}
}  // namespace kernel