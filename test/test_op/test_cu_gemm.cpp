#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"

using namespace kernel;

// CPU reference implementation for GEMM
// C[M,N] = A[M,K] x B[N,K]^T (B stored as [N,K])
void gemm_cpu_reference(const float* A, const float* B, float* C,
                        int M, int K, int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        // A[m,k] * B[n,k]
        sum += A[m * K + k] * B[n * K + k];
      }
      C[m * N + n] = sum;
    }
  }
}

// Test basic GEMM with small matrices
TEST(test_gemm_cu, gemm_basic_small) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int M = 4;  // num_tokens
  const int K = 8;  // in_dim
  const int N = 6;  // out_dim

  // Create input tensors on CPU
  tensor::Tensor input(base::DataType::kDataTypeFp32, M, K, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, N, K, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, M, N, true, alloc_cpu);

  // Initialize with simple values
  for (int i = 0; i < M * K; ++i) {
    input.index<float>(i) = static_cast<float>(i % 7);
  }
  for (int i = 0; i < N * K; ++i) {
    weight.index<float>(i) = static_cast<float>((i % 5) - 2);
  }

  // CPU reference
  gemm_cpu_reference(input.ptr<float>(), weight.ptr<float>(),
                     output_cpu.ptr<float>(), M, K, N);

  // Copy to GPU
  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, M, N, true, alloc_cu);

  // Run GEMM kernel
  // Note: CudaConfig destructor will destroy the stream
  CudaConfig* config = new CudaConfig;
  cudaStreamCreate(&config->stream);
  kernel::get_gemm_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output_cu, config);
  cudaStreamSynchronize(config->stream);

  // Copy back and compare
  output_cu.to_cpu();

  for (int i = 0; i < M * N; ++i) {
    ASSERT_NEAR(output_cu.index<float>(i), output_cpu.index<float>(i), 1e-4f)
        << "Mismatch at index " << i;
  }

  delete config;
}

// Test GEMM with typical LLM dimensions
TEST(test_gemm_cu, gemm_llm_dimensions) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int M = 32;    // num_tokens (small batch for testing)
  const int K = 256;   // in_dim (reduced for testing)
  const int N = 512;   // out_dim

  tensor::Tensor input(base::DataType::kDataTypeFp32, M, K, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, N, K, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, M, N, true, alloc_cpu);

  // Random initialization
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < M * K; ++i) {
    input.index<float>(i) = dist(gen);
  }
  for (int i = 0; i < N * K; ++i) {
    weight.index<float>(i) = dist(gen);
  }

  // CPU reference
  gemm_cpu_reference(input.ptr<float>(), weight.ptr<float>(),
                     output_cpu.ptr<float>(), M, K, N);

  // GPU computation
  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, M, N, true, alloc_cu);

  CudaConfig* config = new CudaConfig;
  cudaStreamCreate(&config->stream);
  kernel::get_gemm_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output_cu, config);
  cudaStreamSynchronize(config->stream);

  output_cu.to_cpu();

  // Compare with tolerance for floating point
  float max_diff = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    float diff = std::abs(output_cu.index<float>(i) - output_cpu.index<float>(i));
    max_diff = std::max(max_diff, diff);
    ASSERT_NEAR(output_cu.index<float>(i), output_cpu.index<float>(i), 1e-3f)
        << "Mismatch at index " << i << ", diff = " << diff;
  }
  LOG(INFO) << "Max difference: " << max_diff;

  delete config;
}

// Test GEMM with single token (should fall back to GEMV)
TEST(test_gemm_cu, gemm_single_token_fallback) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int K = 128;  // in_dim
  const int N = 64;   // out_dim

  // Single token: 1D input
  tensor::Tensor input(base::DataType::kDataTypeFp32, K, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, N, K, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, N, true, alloc_cpu);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < K; ++i) {
    input.index<float>(i) = dist(gen);
  }
  for (int i = 0; i < N * K; ++i) {
    weight.index<float>(i) = dist(gen);
  }

  // CPU reference (M=1)
  gemm_cpu_reference(input.ptr<float>(), weight.ptr<float>(),
                     output_cpu.ptr<float>(), 1, K, N);

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, N, true, alloc_cu);

  CudaConfig* config = new CudaConfig;
  cudaStreamCreate(&config->stream);
  kernel::get_gemm_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output_cu, config);
  cudaStreamSynchronize(config->stream);

  output_cu.to_cpu();

  for (int i = 0; i < N; ++i) {
    ASSERT_NEAR(output_cu.index<float>(i), output_cpu.index<float>(i), 1e-4f)
        << "Mismatch at index " << i;
  }

  delete config;
}

// Test batched GEMV for decode phase
TEST(test_gemm_cu, gemv_batched_decode) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int batch_size = 8;  // num sequences
  const int K = 256;         // in_dim
  const int N = 128;         // out_dim

  tensor::Tensor input(base::DataType::kDataTypeFp32, batch_size, K, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, N, K, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, batch_size, N, true, alloc_cpu);

  std::mt19937 gen(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < batch_size * K; ++i) {
    input.index<float>(i) = dist(gen);
  }
  for (int i = 0; i < N * K; ++i) {
    weight.index<float>(i) = dist(gen);
  }

  // CPU reference
  gemm_cpu_reference(input.ptr<float>(), weight.ptr<float>(),
                     output_cpu.ptr<float>(), batch_size, K, N);

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, batch_size, N, true, alloc_cu);

  CudaConfig* config = new CudaConfig;
  cudaStreamCreate(&config->stream);
  kernel::get_gemv_batched_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output_cu, config);
  cudaStreamSynchronize(config->stream);

  output_cu.to_cpu();

  float max_diff = 0.0f;
  for (int i = 0; i < batch_size * N; ++i) {
    float diff = std::abs(output_cu.index<float>(i) - output_cpu.index<float>(i));
    max_diff = std::max(max_diff, diff);
    ASSERT_NEAR(output_cu.index<float>(i), output_cpu.index<float>(i), 1e-3f)
        << "Mismatch at index " << i;
  }
  LOG(INFO) << "Batched GEMV max difference: " << max_diff;

  delete config;
}

// Test GEMM with non-aligned dimensions (edge cases)
TEST(test_gemm_cu, gemm_non_aligned) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // Non-power-of-2 dimensions
  const int M = 17;   // num_tokens
  const int K = 67;   // in_dim
  const int N = 33;   // out_dim

  tensor::Tensor input(base::DataType::kDataTypeFp32, M, K, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, N, K, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, M, N, true, alloc_cpu);

  std::mt19937 gen(789);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  for (int i = 0; i < M * K; ++i) {
    input.index<float>(i) = dist(gen);
  }
  for (int i = 0; i < N * K; ++i) {
    weight.index<float>(i) = dist(gen);
  }

  gemm_cpu_reference(input.ptr<float>(), weight.ptr<float>(),
                     output_cpu.ptr<float>(), M, K, N);

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, M, N, true, alloc_cu);

  CudaConfig* config = new CudaConfig;
  cudaStreamCreate(&config->stream);
  kernel::get_gemm_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output_cu, config);
  cudaStreamSynchronize(config->stream);

  output_cu.to_cpu();

  for (int i = 0; i < M * N; ++i) {
    ASSERT_NEAR(output_cu.index<float>(i), output_cpu.index<float>(i), 1e-3f)
        << "Mismatch at index " << i;
  }

  delete config;
}

// Test larger GEMM for performance validation
TEST(test_gemm_cu, gemm_large_prefill) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int M = 128;   // num_tokens (prefill)
  const int K = 512;   // in_dim
  const int N = 512;   // out_dim

  tensor::Tensor input(base::DataType::kDataTypeFp32, M, K, true, alloc_cpu);
  tensor::Tensor weight(base::DataType::kDataTypeFp32, N, K, true, alloc_cpu);
  tensor::Tensor output_cpu(base::DataType::kDataTypeFp32, M, N, true, alloc_cpu);

  std::mt19937 gen(999);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < M * K; ++i) {
    input.index<float>(i) = dist(gen);
  }
  for (int i = 0; i < N * K; ++i) {
    weight.index<float>(i) = dist(gen);
  }

  gemm_cpu_reference(input.ptr<float>(), weight.ptr<float>(),
                     output_cpu.ptr<float>(), M, K, N);

  input.to_cuda(nullptr);
  weight.to_cuda(nullptr);

  tensor::Tensor output_cu(base::DataType::kDataTypeFp32, M, N, true, alloc_cu);

  CudaConfig* config = new CudaConfig;
  cudaStreamCreate(&config->stream);

  // Warmup
  kernel::get_gemm_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output_cu, config);
  cudaStreamSynchronize(config->stream);

  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int num_iterations = 10;
  cudaEventRecord(start, config->stream);
  for (int i = 0; i < num_iterations; ++i) {
    kernel::get_gemm_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output_cu, config);
  }
  cudaEventRecord(stop, config->stream);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  LOG(INFO) << "GEMM [" << M << "x" << K << "] x [" << N << "x" << K << "]^T: "
            << milliseconds / num_iterations << " ms per iteration";

  output_cu.to_cpu();

  float max_diff = 0.0f;
  for (int i = 0; i < M * N; ++i) {
    float diff = std::abs(output_cu.index<float>(i) - output_cpu.index<float>(i));
    max_diff = std::max(max_diff, diff);
  }
  LOG(INFO) << "Max difference: " << max_diff;
  ASSERT_LT(max_diff, 1e-2f) << "Results differ too much from CPU reference";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  delete config;
}
