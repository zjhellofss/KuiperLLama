#include <cublas_v2.h>
#include <tensor/tensor.h>
#include "../matmul_kernel_i.h"
#include "matmul_kernel_cu.cuh"
namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale,
                      const CudaConfig* config) {
  CHECK(config != nullptr && config->handle != nullptr);
  cublasHandle_t handle = config->handle;
  if (config->stream) {
    cublasSetStream(handle, config->stream);
  }
  CHECK(input.is_empty() == false && input.dims_size() <= 2);
  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

  CHECK(weight.is_empty() == false && weight.dims_size() == 2);
  CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
  const int32_t M = weight.get_dim(0);
  const int32_t N = weight.get_dim(1);
  int32_t K = 1;
  if (input.dims_size() == 2) {
    K = input.get_dim(1);
  }
  CHECK_EQ(N, input.get_dim(0));

  CHECK(output.is_empty() == false && output.size() == M * K);
  CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);
  float alpha = scale;
  float beta = 0.f;
  auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            K,  // 矩阵B的列数
                            M,  // 矩阵A的行数
                            N,  // 矩阵A的列数
                            &alpha, input.ptr<float>(), K, weight.ptr<float>(), N, &beta,
                            const_cast<float*>(output.ptr<float>()), K);
  CHECK(status == CUBLAS_STATUS_SUCCESS);
}
}  // namespace kernel