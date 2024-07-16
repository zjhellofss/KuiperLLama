#include <tensor/tensor.h>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_kernel.cuh"
namespace kernel {
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M,
                                      int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
#pragma unroll
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    int row_offset = p * M;
#pragma unroll
    for (int i = tid * 4; i < M; i += THREAD_PER_BLOCK * 4) {
      float part_sum = 0.f;
      float4 input_float4 = *(float4*)(input + i);
      float4 weight_float4 = *(float4*)(weight + row_offset + i);
      if (i < M) {
        part_sum += weight_float4.x * input_float4.x;
      }

      if (i + 1 < M) {
        part_sum += weight_float4.y * input_float4.y;
      }

      if (i + 2 < M) {
        part_sum += weight_float4.z * input_float4.z;
      }

      if (i + 3 < M) {
        part_sum += weight_float4.w * input_float4.w;
      }
      sdata[tid] += part_sum;
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32int8(const float* input, const int8_t* weight,
                                          const float* scales, const int32_t group_size,
                                          float* output, int M, int K) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  unsigned int tid = threadIdx.x;

  int start_row = blockIdx.x * ROW_PER_BLOCK;
  int end_row = start_row + ROW_PER_BLOCK;
  if (start_row >= K) {
    return;
  }
  for (int p = start_row; p < end_row; ++p) {
    sdata[tid] = 0;
    for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
      const int weight_idx = p * M + i;
      const int group_idx = weight_idx / group_size;
      sdata[tid] += input[i] * scales[group_idx] * static_cast<float>(weight[weight_idx]);
    }
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    float part_sum = BlockReduce(temp).Sum(sdata[tid]);
    __syncthreads();

    if (tid == 0) {
      output[p] = part_sum;
    }
    __syncthreads();
  }
}

void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  if (config->stream) {
  }
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  CHECK_EQ(M, input.get_dim(0));
  matmul_kernel_cu_fp32<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<float>(),
                                            const_cast<float*>(output.ptr<float>()), M, K);
}

void matmul_kernel_cu_qint8(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const CudaConfig* config) {
  CHECK(config != nullptr);
  if (config->stream) {
  }
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  CHECK_EQ(M, input.get_dim(0));
  matmul_kernel_cu_fp32int8<128, 1><<<K, 128>>>(input.ptr<float>(), weight.ptr<int8_t>(),
                                                scale.ptr<float>(), group_size,
                                                const_cast<float*>(output.ptr<float>()), M, K);
}
}  // namespace kernel