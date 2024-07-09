#include "rope_kernel.cuh"
namespace kernel {

__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) {
  float v0 = vec[idx];
  float v1 = vec[idx + 1];
  vec[idx] = v0 * fcr - v1 * fci;
  vec[idx + 1] = v0 * fci + v1 * fcr;
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx = idx * 2;
  if (idx >= dim) {
    return;
  }

  int head_dim = idx % head_size;
  float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
  float val = static_cast<float>(pos) * freq;
  float fcr = cosf(val);
  float fci = sinf(val);
  rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
  if (idx >= kv_dim) {
    return;
  }
  rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos, void* stream) {
  const int32_t pos = *input_pos.ptr<int32_t>(0);
  int threads = 512;
  int blocks = (dim + threads - 1) / threads;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>());
  } else {
    rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                             input_k.ptr<float>());
  }
}
}  // namespace kernel