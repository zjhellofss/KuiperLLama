#ifndef GEMM_KERNEL_CU_CUH
#define GEMM_KERNEL_CU_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"

namespace kernel {

/**
 * Tiled GEMM kernel for multi-token matrix multiplication
 *
 * Computes: C[M, N] = A[M, K] × B[K, N]^T
 * Where B is stored as [N, K] (row-major, transposed)
 *
 * For LLM inference:
 *   A = Input    [num_tokens, in_dim]
 *   B = Weight   [out_dim, in_dim]  (stored transposed)
 *   C = Output   [num_tokens, out_dim]
 *
 * @param input   Input tensor [num_tokens, in_dim] or [in_dim] for single token
 * @param weight  Weight tensor [out_dim, in_dim]
 * @param output  Output tensor [num_tokens, out_dim] or [out_dim] for single token
 * @param config  CUDA configuration (stream, etc.)
 */
void gemm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, const CudaConfig* config = nullptr);

/**
 * Batched GEMV kernel for multi-sequence decode phase
 * Each sequence has exactly 1 token
 *
 * @param input   Input tensor [batch_size, in_dim]
 * @param weight  Weight tensor [out_dim, in_dim]
 * @param output  Output tensor [batch_size, out_dim]
 * @param config  CUDA configuration
 */
void gemv_batched_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, const CudaConfig* config = nullptr); 
                            

}  // namespace kernel

#endif  // GEMM_KERNEL_CU_CUH
