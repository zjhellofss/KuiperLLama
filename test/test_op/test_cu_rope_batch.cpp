#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "../source/op/kernels/kernels_interface.h"
#include "../source/op/kernels/cuda/rope_kernel.cuh"
#include "base/buffer.h"

using namespace kernel;

// Test batched RoPE kernel
TEST(test_rope_batch_cu, rope_batch_basic) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int num_tokens = 4;
  const int num_heads = 8;
  const int head_size = 64;
  const int dim = num_heads * head_size;      // 512
  const int num_kv_heads = 4;
  const int kv_dim = num_kv_heads * head_size; // 256
  const int max_seq_len = 128;

  // Create tensors
  tensor::Tensor input_q(base::DataType::kDataTypeFp32, num_tokens, dim, true, alloc_cpu);
  tensor::Tensor input_k(base::DataType::kDataTypeFp32, num_tokens, kv_dim, true, alloc_cpu);
  tensor::Tensor positions(base::DataType::kDataTypeInt32, num_tokens, true, alloc_cpu);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);

  // Initialize Q and K with random values
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < num_tokens * dim; ++i) {
    input_q.index<float>(i) = dist(gen);
  }
  for (int i = 0; i < num_tokens * kv_dim; ++i) {
    input_k.index<float>(i) = dist(gen);
  }

  // Set positions: [0, 1, 2, 3]
  for (int i = 0; i < num_tokens; ++i) {
    positions.index<int32_t>(i) = i;
  }

  // Clone for comparison
  tensor::Tensor input_q_ref = input_q.clone();
  tensor::Tensor input_k_ref = input_k.clone();

  // Move to GPU
  input_q.to_cuda(nullptr);
  input_k.to_cuda(nullptr);
  positions.to_cuda(nullptr);
  sin_cache.to_cuda(nullptr);
  cos_cache.to_cuda(nullptr);

  // Calculate sin/cos cache on GPU
  CudaConfig* config = new CudaConfig;
  cudaStreamCreate(&config->stream);

  kernel::sin_cos_cache_calc_cu(head_size, max_seq_len, sin_cache, cos_cache, config->stream);
  cudaStreamSynchronize(config->stream);

  // Run batched RoPE kernel
  kernel::get_rope_batch_kernel(base::DeviceType::kDeviceCUDA)(
      num_tokens, dim, kv_dim, head_size,
      input_q, input_k, positions, sin_cache, cos_cache, config->stream);
  cudaStreamSynchronize(config->stream);

  // Copy results back
  input_q.to_cpu();
  input_k.to_cpu();

  // Compute reference using single-token RoPE kernel for each token
  for (int t = 0; t < num_tokens; ++t) {
    // Create single-token tensors
    tensor::Tensor q_single(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
    tensor::Tensor k_single(base::DataType::kDataTypeFp32, kv_dim, true, alloc_cpu);
    // pos_single must stay on CPU because rope_kernel_cu reads it on host
    tensor::Tensor pos_single(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);

    // Copy data
    for (int i = 0; i < dim; ++i) {
      q_single.index<float>(i) = input_q_ref.index<float>(t * dim + i);
    }
    for (int i = 0; i < kv_dim; ++i) {
      k_single.index<float>(i) = input_k_ref.index<float>(t * kv_dim + i);
    }
    pos_single.index<int32_t>(0) = t;

    // Move Q and K to GPU (pos stays on CPU!)
    q_single.to_cuda(nullptr);
    k_single.to_cuda(nullptr);

    // Run single-token RoPE
    kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(
        dim, kv_dim, head_size, q_single, k_single, pos_single,
        sin_cache, cos_cache, config->stream);
    cudaStreamSynchronize(config->stream);

    // Copy back
    q_single.to_cpu();
    k_single.to_cpu();

    // Compare Q
    for (int i = 0; i < dim; ++i) {
      ASSERT_NEAR(input_q.index<float>(t * dim + i), q_single.index<float>(i), 1e-5f)
          << "Q mismatch at token " << t << ", index " << i;
    }

    // Compare K
    for (int i = 0; i < kv_dim; ++i) {
      ASSERT_NEAR(input_k.index<float>(t * kv_dim + i), k_single.index<float>(i), 1e-5f)
          << "K mismatch at token " << t << ", index " << i;
    }
  }

  LOG(INFO) << "Batched RoPE test passed for " << num_tokens << " tokens";
  delete config;
}

// Test batched RoPE with non-sequential positions (for multi-sequence decode)
TEST(test_rope_batch_cu, rope_batch_multi_sequence) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  const int num_tokens = 4;  // 4 sequences, each with 1 token
  const int num_heads = 8;
  const int head_size = 64;
  const int dim = num_heads * head_size;
  const int num_kv_heads = 4;
  const int kv_dim = num_kv_heads * head_size;
  const int max_seq_len = 256;

  tensor::Tensor input_q(base::DataType::kDataTypeFp32, num_tokens, dim, true, alloc_cpu);
  tensor::Tensor input_k(base::DataType::kDataTypeFp32, num_tokens, kv_dim, true, alloc_cpu);
  tensor::Tensor positions(base::DataType::kDataTypeInt32, num_tokens, true, alloc_cpu);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, alloc_cpu);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < num_tokens * dim; ++i) {
    input_q.index<float>(i) = dist(gen);
  }
  for (int i = 0; i < num_tokens * kv_dim; ++i) {
    input_k.index<float>(i) = dist(gen);
  }

  // Non-sequential positions (simulating different sequences at different positions)
  // Seq 0 at pos 10, Seq 1 at pos 25, Seq 2 at pos 5, Seq 3 at pos 100
  int pos_values[] = {10, 25, 5, 100};
  positions.index<int32_t>(0) = pos_values[0];
  positions.index<int32_t>(1) = pos_values[1];
  positions.index<int32_t>(2) = pos_values[2];
  positions.index<int32_t>(3) = pos_values[3];

  tensor::Tensor input_q_ref = input_q.clone();
  tensor::Tensor input_k_ref = input_k.clone();

  input_q.to_cuda(nullptr);
  input_k.to_cuda(nullptr);
  positions.to_cuda(nullptr);
  sin_cache.to_cuda(nullptr);
  cos_cache.to_cuda(nullptr);

  CudaConfig* config = new CudaConfig;
  cudaStreamCreate(&config->stream);

  kernel::sin_cos_cache_calc_cu(head_size, max_seq_len, sin_cache, cos_cache, config->stream);
  cudaStreamSynchronize(config->stream);

  kernel::get_rope_batch_kernel(base::DeviceType::kDeviceCUDA)(
      num_tokens, dim, kv_dim, head_size,
      input_q, input_k, positions, sin_cache, cos_cache, config->stream);
  cudaStreamSynchronize(config->stream);

  input_q.to_cpu();
  input_k.to_cpu();

  // Verify each token independently
  for (int t = 0; t < num_tokens; ++t) {
    tensor::Tensor q_single(base::DataType::kDataTypeFp32, dim, true, alloc_cpu);
    tensor::Tensor k_single(base::DataType::kDataTypeFp32, kv_dim, true, alloc_cpu);
    // pos_single must stay on CPU!
    tensor::Tensor pos_single(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);

    for (int i = 0; i < dim; ++i) {
      q_single.index<float>(i) = input_q_ref.index<float>(t * dim + i);
    }
    for (int i = 0; i < kv_dim; ++i) {
      k_single.index<float>(i) = input_k_ref.index<float>(t * kv_dim + i);
    }
    pos_single.index<int32_t>(0) = pos_values[t];

    q_single.to_cuda(nullptr);
    k_single.to_cuda(nullptr);
    // pos_single stays on CPU

    kernel::get_rope_kernel(base::DeviceType::kDeviceCUDA)(
        dim, kv_dim, head_size, q_single, k_single, pos_single,
        sin_cache, cos_cache, config->stream);
    cudaStreamSynchronize(config->stream);

    q_single.to_cpu();
    k_single.to_cpu();

    for (int i = 0; i < dim; ++i) {
      ASSERT_NEAR(input_q.index<float>(t * dim + i), q_single.index<float>(i), 1e-5f)
          << "Q mismatch at token " << t << " (pos=" << pos_values[t] << "), index " << i;
    }
    for (int i = 0; i < kv_dim; ++i) {
      ASSERT_NEAR(input_k.index<float>(t * kv_dim + i), k_single.index<float>(i), 1e-5f)
          << "K mismatch at token " << t << " (pos=" << pos_values[t] << "), index " << i;
    }
  }

  LOG(INFO) << "Multi-sequence batched RoPE test passed";
  delete config;
}
