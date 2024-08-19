#ifdef USE_CUTLASS

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemv.h>
#include <cutlass/gemm/kernel/gemv.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/device_memory.h>
#include <tensor/tensor.h>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include "../kernels_interface.h"
#include "matmul_cutlass.cuh"
namespace kernel {
void matmul_cutlass(const tensor::Tensor& input, const tensor::Tensor& weight,
                    const tensor::Tensor& output, const float scale, const CudaConfig* config) {
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t K = weight.get_dim(0);  // row
  const int32_t M = weight.get_dim(1);  // col
  int packet_size = 4;
  CHECK_EQ(M % packet_size, 0);
  CHECK_EQ(M, input.get_dim(0));

  using ElementInput = float;
  using ElementOutput = float;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  const int kElementsPerAccess = 4;
  const int kThreadCount = 128;
  const int kThreadPerRow = 32;

  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombination<ElementOutput, 1, ElementAccumulator,
                                                   ElementAccumulator>;
  using Gemv = cutlass::gemm::device::Gemv<
      cutlass::gemm::kernel::Gemv<ElementInput,        // Element A
                                  LayoutA,             // Layout A
                                  ElementInput,        // Element B
                                  ElementOutput,       // Element C
                                  ElementAccumulator,  // Element accumulator
                                  EpilogueOp,          // Output operator
                                  kElementsPerAccess,  // Element access granularity
                                  kThreadCount,        // Thread count
                                  kThreadPerRow        // Threads per row
                                  >>;

  typename Gemv::Arguments arguments{{K, M},
                                     1,
                                     {scale, 0},
                                     {const_cast<float*>(weight.ptr<float>()), M},
                                     input.ptr<float>(),
                                     output.ptr<float>(),
                                     const_cast<float*>(output.ptr<float>()),
                                     static_cast<int64_t>(K) * static_cast<int64_t>(M),
                                     static_cast<int64_t>(M),
                                     static_cast<int64_t>(K),
                                     static_cast<int64_t>(K)};
  Gemv gemv_op;

  cutlass::Status status = gemv_op.can_implement(arguments);
  CHECK(status == cutlass::Status::kSuccess);

  size_t workspace_size = gemv_op.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  if (config && config->stream) {
    status = gemv_op.initialize(arguments, workspace.get(), config->stream);
  } else {
    status = gemv_op.initialize(arguments, workspace.get());
  }
  CHECK(status == cutlass::Status::kSuccess);

  status = gemv_op();
  CHECK(status == cutlass::Status::kSuccess);
}
}  // namespace kernel

#endif  // USE_CUTLASS