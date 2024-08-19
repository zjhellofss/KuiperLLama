#include <base/base.h>
#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/scale_sum_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cuda/emb_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "cuda/matmul_cutlass.cuh"
#include "cuda/mha_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/rope_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#include "kernels_interface.h"
namespace kernel {
AddKernel get_add_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return add_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return add_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return emb_kernel_normal;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return emb_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an embedding kernel.";
    return nullptr;
  }
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return matmul_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
#ifdef USE_CUTLASS
    return matmul_cutlass;
#else
    return matmul_kernel_cu;
#endif
  } else {
    LOG(FATAL) << "Unknown device type for get an matmul kernel.";
    return nullptr;
  }
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCUDA) {
    return matmul_kernel_cu_qint8;
  } else {
    LOG(FATAL) << "Unknown device type for get an matmul kernel.";
    return nullptr;
  }
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return mha_kernel;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return mha_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an mha kernel.";
    return nullptr;
  }
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rope_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rope_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}

ScaleKernel get_scale_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return scale_inplace_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return softmax_inplace_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get an softmax kernel.";
    return nullptr;
  }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return swiglu_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return swiglu_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
    return nullptr;
  }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == base::DeviceType::kDeviceCUDA) {
    return rmsnorm_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get an rmsnorm kernel.";
    return nullptr;
  }
}

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
  if (device_type == base::DeviceType::kDeviceCPU) {
    return scale_sum_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a scale and reduce kernel.";
    return nullptr;
  }
}

}  // namespace kernel
