#include <cuda_runtime.h>
#include <glog/logging.h>
#include "gemm_kernel.cuh"
#include "matmul_kernel.cuh"

namespace kernel {

// ============================================================================
// Tiled GEMM Kernel Configuration
// ============================================================================
// Block tile size: each thread block computes BLOCK_M x BLOCK_N output elements
// Thread tile size: each thread computes THREAD_M x THREAD_N output elements
// K dimension is processed in chunks of BLOCK_K

constexpr int BLOCK_M = 64;     // Block tile size in M dimension
constexpr int BLOCK_N = 64;     // Block tile size in N dimension
constexpr int BLOCK_K = 32;     // Block tile size in K dimension
constexpr int THREAD_M = 4;     // Thread tile size in M dimension
constexpr int THREAD_N = 4;     // Thread tile size in N dimension

// Derived constants
constexpr int THREADS_M = BLOCK_M / THREAD_M;  // 16 threads in M dimension
constexpr int THREADS_N = BLOCK_N / THREAD_N;  // 16 threads in N dimension
constexpr int THREADS_PER_BLOCK = THREADS_M * THREADS_N;  // 256 threads total

// ============================================================================
// Tiled GEMM Kernel
// ============================================================================
// Computes C[M,N] = A[M,K] x B[N,K]^T where B is stored as [N,K]
//
// Memory Layout:
//   A: row-major [M, K]
//   B: row-major [N, K] (logically transposed, so B[n,k] accesses weight[n*K+k])
//   C: row-major [M, N]
//
// Optimization techniques:
//   1. Shared memory tiling to reduce global memory access
//   2. Thread-level tiling for register reuse
//   3. Vectorized loads (float4) where possible
//   4. Loop unrolling for better instruction-level parallelism

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [N, K] - weight matrix (transposed storage)
    float* __restrict__ C,        // [M, N]
    const int M, const int K, const int N) {

  // Block and thread indices
  const int bx = blockIdx.x;   // Block index in N dimension
  const int by = blockIdx.y;   // Block index in M dimension
  const int tx = threadIdx.x;  // Thread index in N dimension (0 to THREADS_N-1)
  const int ty = threadIdx.y;  // Thread index in M dimension (0 to THREADS_M-1)

  // Starting position of this block's output tile in C
  const int c_row_start = by * BM;
  const int c_col_start = bx * BN;

  // Shared memory for A and B tiles
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // Register accumulator for this thread's output tile (TM x TN elements)
  float c_reg[TM][TN] = {{0.0f}};

  // This thread's position within the block's output tile
  const int thread_row = ty * TM;  // Starting row in block tile
  const int thread_col = tx * TN;  // Starting col in block tile

  // Linear thread ID for cooperative loading
  const int tid = ty * (BN / TN) + tx;
  constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

  // Calculate how many elements each thread loads for A and B tiles
  constexpr int A_TILE_ELEMENTS = BM * BK;
  constexpr int B_TILE_ELEMENTS = BK * BN;
  constexpr int A_LOADS_PER_THREAD = (A_TILE_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;
  constexpr int B_LOADS_PER_THREAD = (B_TILE_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;

  // Iterate over K dimension in chunks of BK
  for (int k_start = 0; k_start < K; k_start += BK) {

    // ========== Load A tile into shared memory ==========
    // A tile: [BM, BK] from A[c_row_start:c_row_start+BM, k_start:k_start+BK]
    #pragma unroll
    for (int load_idx = 0; load_idx < A_LOADS_PER_THREAD; ++load_idx) {
      int elem_idx = tid + load_idx * NUM_THREADS;
      if (elem_idx < A_TILE_ELEMENTS) {
        int a_tile_row = elem_idx / BK;
        int a_tile_col = elem_idx % BK;
        int global_row = c_row_start + a_tile_row;
        int global_col = k_start + a_tile_col;

        if (global_row < M && global_col < K) {
          As[a_tile_row][a_tile_col] = A[global_row * K + global_col];
        } else {
          As[a_tile_row][a_tile_col] = 0.0f;
        }
      }
    }

    // ========== Load B tile into shared memory ==========
    // B is stored as [N, K], we load B[c_col_start:c_col_start+BN, k_start:k_start+BK]
    // and store it transposed as Bs[BK, BN] for coalesced access during computation
    #pragma unroll
    for (int load_idx = 0; load_idx < B_LOADS_PER_THREAD; ++load_idx) {
      int elem_idx = tid + load_idx * NUM_THREADS;
      if (elem_idx < B_TILE_ELEMENTS) {
        int b_tile_row = elem_idx / BN;  // K dimension (0 to BK-1)
        int b_tile_col = elem_idx % BN;  // N dimension (0 to BN-1)
        int global_k = k_start + b_tile_row;
        int global_n = c_col_start + b_tile_col;

        if (global_k < K && global_n < N) {
          // B[N, K] -> Bs[K, N]: B[global_n, global_k] = B[global_n * K + global_k]
          Bs[b_tile_row][b_tile_col] = B[global_n * K + global_k];
        } else {
          Bs[b_tile_row][b_tile_col] = 0.0f;
        }
      }
    }

    __syncthreads();

    // ========== Compute partial results ==========
    // Each thread computes its TM x TN output elements
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      // Load A values for this thread's rows into registers
      float a_reg[TM];
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        a_reg[m] = As[thread_row + m][k];
      }

      // Load B values for this thread's columns into registers
      float b_reg[TN];
      #pragma unroll
      for (int n = 0; n < TN; ++n) {
        b_reg[n] = Bs[k][thread_col + n];
      }

      // Outer product accumulation
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
          c_reg[m][n] += a_reg[m] * b_reg[n];
        }
      }
    }

    __syncthreads();
  }

  // ========== Write results to global memory ==========
  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    #pragma unroll
    for (int n = 0; n < TN; ++n) {
      int global_row = c_row_start + thread_row + m;
      int global_col = c_col_start + thread_col + n;

      if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = c_reg[m][n];
      }
    }
  }
}

// ============================================================================
// Batched GEMV Kernel for Decode Phase
// ============================================================================
// For decode phase: each sequence has 1 token, but we have multiple sequences
// Input: [batch_size, in_dim], Weight: [out_dim, in_dim], Output: [batch_size, out_dim]

template <int THREADS>
__global__ void gemv_batched_kernel(
    const float* __restrict__ input,   // [batch_size, in_dim]
    const float* __restrict__ weight,  // [out_dim, in_dim]
    float* __restrict__ output,        // [batch_size, out_dim]
    const int batch_size,
    const int in_dim,
    const int out_dim) {

  // Grid: (out_dim, batch_size) - each block computes one output element for one sequence
  const int out_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;

  if (batch_idx >= batch_size || out_idx >= out_dim) return;

  // Pointers to this sequence's input and weight row
  const float* x = input + batch_idx * in_dim;
  const float* w = weight + out_idx * in_dim;

  // Shared memory for partial sums
  __shared__ float sdata[THREADS];
  float sum = 0.0f;

  // Vectorized load using float4 for better memory bandwidth
  const int vec_size = 4;
  const int vec_num = in_dim / vec_size;
  const float4* x_vec = reinterpret_cast<const float4*>(x);
  const float4* w_vec = reinterpret_cast<const float4*>(w);

  // Process vectorized portion
  for (int i = tid; i < vec_num; i += THREADS) {
    float4 xv = x_vec[i];
    float4 wv = w_vec[i];
    sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
  }

  // Process remaining elements
  const int vec_offset = vec_num * vec_size;
  for (int i = vec_offset + tid; i < in_dim; i += THREADS) {
    sum += x[i] * w[i];
  }

  sdata[tid] = sum;
  __syncthreads();

  // Block reduction
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result
  if (tid == 0) {
    output[batch_idx * out_dim + out_idx] = sdata[0];
  }
}

// ============================================================================
// Public Interface Functions
// ============================================================================

void gemm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, const CudaConfig* config) {
  CHECK(!input.is_empty()) << "Input tensor is empty";
  CHECK(!weight.is_empty()) << "Weight tensor is empty";
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA) << "Input must be on CUDA";
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA) << "Weight must be on CUDA";

  // Determine dimensions
  // Input can be [num_tokens, in_dim] or [in_dim] for single token
  int M, K, N;
  if (input.dims_size() == 1) {
    // Single token case: [in_dim]
    M = 1;
    K = input.get_dim(0);
  } else {
    // Multi-token case: [num_tokens, in_dim]
    M = input.get_dim(0);
    K = input.get_dim(1);
  }

  // Weight is [out_dim, in_dim]
  N = weight.get_dim(0);
  CHECK_EQ(weight.get_dim(1), K) << "Weight dimension mismatch";

  // For single token with 1D input, fall back to existing optimized GEMV
  // Note: only fallback when input is 1D [dim], not 2D [1, dim]
  if (M == 1 && input.dims_size() == 1) {
    matmul_kernel_cu(input, weight, output, 1.0f, config);
    return;
  }

  // Multi-token: use tiled GEMM
  dim3 block(THREADS_N, THREADS_M);  // (16, 16) = 256 threads
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  cudaStream_t stream = config ? config->stream : nullptr;

  gemm_tiled_kernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N>
      <<<grid, block, 0, stream>>>(
          input.ptr<float>(),
          weight.ptr<float>(),
          const_cast<float*>(output.ptr<float>()),
          M, K, N);
}

void gemv_batched_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, const CudaConfig* config) {
  CHECK(!input.is_empty()) << "Input tensor is empty";
  CHECK(!weight.is_empty()) << "Weight tensor is empty";
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA) << "Input must be on CUDA";
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA) << "Weight must be on CUDA";
  CHECK_EQ(input.dims_size(), 2) << "Input must be 2D [batch_size, in_dim]";

  const int batch_size = input.get_dim(0);
  const int in_dim = input.get_dim(1);
  const int out_dim = weight.get_dim(0);

  CHECK_EQ(weight.get_dim(1), in_dim) << "Weight dimension mismatch";

  // For small batch sizes, use custom batched GEMV kernel
  // For larger batches, GEMM might be more efficient
  constexpr int BATCH_THRESHOLD = 16;

  if (batch_size <= BATCH_THRESHOLD) {
    constexpr int THREADS = 256;
    dim3 grid(out_dim, batch_size);
    dim3 block(THREADS);

    cudaStream_t stream = config ? config->stream : nullptr;

    gemv_batched_kernel<THREADS><<<grid, block, 0, stream>>>(
        input.ptr<float>(),
        weight.ptr<float>(),
        const_cast<float*>(output.ptr<float>()),
        batch_size, in_dim, out_dim);
  } else {
    // Fall back to GEMM for larger batches
    gemm_kernel_cu(input, weight, output, config);
  }
}

}  // namespace kernel
