#include "model/llama3.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <cmath>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include <sentencepiece_processor.h>
#include <utility>
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "../op/kernels/cuda/paged_attention_kernel.cuh"
#include "../op/kernels/kernels_interface.h"
#include "base/tick.h"
namespace model {

void LLama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    embedding_layer_->set_cuda_config(config);
    embedding_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->to_cuda();
      rms_norm_layer->set_cuda_config(config);
    }
  }
}

LLama2Model::LLama2Model(base::TokenizerType tokenizer_type, std::string token_path,
                         std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {}

base::Status LLama2Model::init(base::DeviceType device_type) {
  using namespace base;
  if (token_path_.empty()) {
    return error::PathNotValid(token_path_);
  }
  if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return error::InternalError("The cpu device do not support int8 quant model.");
  }

  device_type_ = device_type;
  if (device_type == DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return error::InternalError("The cuda hanle create failed.");
    }
  }

  Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }
  init_mem();
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
  return error::Success();
}

base::Status LLama2Model::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  const bool use_paged_attention = (kv_cache_manager_ != nullptr);

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    // attention (wq wk wv @ input)
    attention_qkv(layer_idx, pos_tensor);
    // multi-head attention
    if (use_paged_attention) {
      attention_mha_paged(layer_idx, pos_tensor);
    } else {
      attention_mha(layer_idx, pos_tensor);
    }
    // feed forward
    feed_forward(layer_idx, input);
  }
  cls_logits(input);
  return base::error::Success();
}

void LLama2Model::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
      device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
      device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
      config_->head_size_);

  llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

  llama_layers_->swiglu_layer_ =
      std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

void LLama2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                          cpu_device_type);
    pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
  }
  llama_layers_->cls_layer_ = cls_layer;

  // embedding layer
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                              cpu_device_type);
  weight_ptr += config_->vocab_size_ * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

void LLama2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);
  // The embedding layer
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

  const void* weight_embedding = raw_model_data_->weight(0);
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                              weight_embedding, cpu_device_type);

  // create all matmul layer
  int32_t dim = config_->dim_;
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
  // create weight matrix for query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;
  }

  // create weight matrix for key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // create weight matrix for output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip ffn rmsnorm
  pos += config_->layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip final rms weight
  pos += dim;
  // skip freqs_cos and freqs_sin weight
  pos += config_->seq_len_ * config_->head_size_;

  llama_layers_->cls_layer_ =
      std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(0), cpu_device_type);
  } else {
    llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                          this->raw_model_data_->weight(pos), cpu_device_type);
  }

  // create rmsnorm layer
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }

  // skip attention.wq attention.wk attention.wv attention.wo
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos +=
      config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  // skip ffn.w1 ffn.w2 ffn.w3
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  std::shared_ptr<op::RmsNormLayer> rms_final_layer =
      std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

void LLama2Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    llama_layers_->to_cuda(cuda_config_);
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
      base::CUDADeviceAllocatorFactory::get_instance();

  tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
  tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);
  tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                           true, alloc);

  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  // kv cache
  tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);
  tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);

  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

  // Wq query output
  tensor::Tensor query(base::DataType::kDataTypeFp32, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));

  // Pos tensor
  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  // Attention output
  tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                      alloc);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  // final forward output
  tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                      alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status LLama2Model::create_layers() {
  using namespace base;
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<LLama2Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }
  create_nonparam_layers();

  if (!llama_layers_->embedding_layer_) {
    return error::InternalError("Create the embedding layer for the llama model failed!");
  }

  if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return error::InternalError("Create the rmsnorm layers for the llama model failed!");
  }

  if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
      llama_layers_->wk_layers_.size() != config_->layer_num_ ||
      llama_layers_->wv_layers_.size() != config_->layer_num_ ||
      llama_layers_->wo_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for "
        "the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
        !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the attention and ffn attention layers for "
          "the llama model "
          "failed.");
    }
  }

  if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
      llama_layers_->w2_layers_.size() != config_->layer_num_ ||
      llama_layers_->w3_layers_.size() != config_->layer_num_) {
    return error::InternalError(
        "Create the matmul layer in the feedforward layers for the llama model "
        "failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
        !llama_layers_->w3_layers_.at(i)) {
      return error::InternalError(
          "Create the matmul layer in the feedforward layers for the llama model "
          "failed.");
    }
  }

  if (!llama_layers_->rope_layer_) {
    return error::InternalError("Create the rope layer for the llama model failed!");
  }

  if (!llama_layers_->add_layer_) {
    return error::InternalError("Create the add layer for the llama model failed!");
  }

  if (!llama_layers_->mha_layer_) {
    return error::InternalError("Create the mha layer for the llama model failed!");
  }

  if (!llama_layers_->swiglu_layer_) {
    return error::InternalError("Create the SwiGLU layer for the llama model failed!");
  }
  return error::Success();
}

op::EmbeddingOutput LLama2Model::embedding(const std::vector<int>& tokens) const {
  auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
  auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
  }
  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  auto input_token_num =
      tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !llama_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(
      llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
  return output;
}

void LLama2Model::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // attn rmsnorm
  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  // value
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

base::Status LLama2Model::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                  bool is_prompt, int& next) const {
  auto status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }
  next = post_processing(pos_tensor, is_prompt);
  return base::error::Success();
}

void LLama2Model::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // mha
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  // VAL = [val1,val2,...val t]
  // output @ VAL = 最终的结果
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  const auto& mha_layer = llama_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LLama2Model::attention_mha_paged(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  CHECK(kv_cache_manager_ != nullptr) << "KV cache manager is null for paged attention";

  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);

  // Get paged KV cache tensors
  const tensor::Tensor& kv_cache = kv_cache_manager_->kv_cache_device();
  const tensor::Tensor& block_tables = kv_cache_manager_->block_tables_device();
  const tensor::Tensor& context_lens = kv_cache_manager_->context_lens_device();

  // Get parameters from config and kv_cache_manager
  const int32_t num_heads = config_->head_num_;
  const int32_t num_kv_heads = config_->kv_head_num_;
  const int32_t head_dim = config_->head_size_;
  const int32_t num_layers = config_->layer_num_;
  const int32_t block_size = kv_cache_manager_->block_size();
  const int32_t max_blocks_per_seq = kv_cache_manager_->max_blocks_per_seq();
  // Get num_seqs from KVCacheManager instead of hardcoding
  const int32_t num_seqs = kv_cache_manager_->num_seqs();
  const int32_t max_context_len = kv_cache_manager_->get_context_len(0);

  // Scale factor for attention: 1/sqrt(head_dim)
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Get CUDA stream
  cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;

  // Call paged attention kernel
  kernel::paged_attention_kernel_cu(
      mha_output,
      query,
      kv_cache,
      block_tables,
      context_lens,
      max_context_len,
      num_seqs,
      num_heads,
      num_kv_heads,
      head_dim,
      block_size,
      max_blocks_per_seq,
      scale,
      layer_idx,
      num_layers,
      stream);

  // wo @ attention output
  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LLama2Model::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(
      llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
      << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
      << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
      << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}

void LLama2Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(llama_layers_->cls_layer_, nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

int32_t LLama2Model::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
  }
  return next;
}

void LLama2Model::init_batch_buffers(int32_t num_seqs) {
  if (num_seqs <= 1) {
    return;  // No need for batch buffers with single sequence
  }

  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }
  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
      base::CPUDeviceAllocatorFactory::get_instance();

  // Batch input embeddings: [num_seqs, dim]
  tensor::Tensor batch_input(base::DataType::kDataTypeFp32, num_seqs, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchInputEmbeddings, batch_input));

  // Batch positions: [num_seqs] on CPU (for reading), and GPU copy
  tensor::Tensor batch_positions_cpu(base::DataType::kDataTypeInt32, num_seqs, true, alloc_cpu);
  tensor::Tensor batch_positions_gpu(base::DataType::kDataTypeInt32, num_seqs, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchPositions, batch_positions_gpu));

  // Batch RMSNorm output: [num_seqs, dim]
  tensor::Tensor batch_rms_output(base::DataType::kDataTypeFp32, num_seqs, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchRMSNormOutput, batch_rms_output));

  // Batch Query: [num_seqs, dim]
  tensor::Tensor batch_query(base::DataType::kDataTypeFp32, num_seqs, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchQuery, batch_query));

  // Batch Key: [num_seqs, kv_dim]
  tensor::Tensor batch_key(base::DataType::kDataTypeFp32, num_seqs, config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchKey, batch_key));

  // Batch Value: [num_seqs, kv_dim]
  tensor::Tensor batch_value(base::DataType::kDataTypeFp32, num_seqs, config_->kv_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchValue, batch_value));

  // Batch MHA output: [num_seqs, dim]
  tensor::Tensor batch_mha_output(base::DataType::kDataTypeFp32, num_seqs, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchOutputMHA, batch_mha_output));

  // Batch Attention output: [num_seqs, dim]
  tensor::Tensor batch_attn_output(base::DataType::kDataTypeFp32, num_seqs, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchAttnOutput, batch_attn_output));

  // Batch FFN RMSNorm: [num_seqs, dim]
  tensor::Tensor batch_ffn_rms(base::DataType::kDataTypeFp32, num_seqs, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchFFNRMSNorm, batch_ffn_rms));

  // Batch W1 output: [num_seqs, hidden_dim]
  tensor::Tensor batch_w1_output(base::DataType::kDataTypeFp32, num_seqs, config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchW1Output, batch_w1_output));

  // Batch W3 output: [num_seqs, hidden_dim]
  tensor::Tensor batch_w3_output(base::DataType::kDataTypeFp32, num_seqs, config_->hidden_dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchW3Output, batch_w3_output));

  // Batch W2 output: [num_seqs, dim]
  tensor::Tensor batch_w2_output(base::DataType::kDataTypeFp32, num_seqs, config_->dim_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchW2Output, batch_w2_output));

  // Batch forward output: [num_seqs, vocab_size] - logits for each sequence
  tensor::Tensor batch_forward_output(base::DataType::kDataTypeFp32, num_seqs, config_->vocab_size_, true, alloc);
  CHECK(insert_buffer(ModelBufferType::kBatchForwardOutput, batch_forward_output));

  LOG(INFO) << "Initialized batch buffers for " << num_seqs << " sequences";
}

base::Status LLama2Model::forward_batch(const tensor::Tensor& inputs,
                                        const tensor::Tensor& positions,
                                        std::vector<int32_t>& next_tokens) const {
  // inputs: [num_seqs, dim]
  // positions: [num_seqs] on GPU
  if (inputs.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }

  const int32_t num_seqs = kv_cache_manager_->num_seqs();
  if (num_seqs <= 1) {
    return base::error::InvalidArgument("forward_batch requires num_seqs > 1");
  }

  // Get batch buffers
  tensor::Tensor batch_rms_output = get_buffer(ModelBufferType::kBatchRMSNormOutput);
  tensor::Tensor batch_query = get_buffer(ModelBufferType::kBatchQuery);
  tensor::Tensor batch_key = get_buffer(ModelBufferType::kBatchKey);
  tensor::Tensor batch_value = get_buffer(ModelBufferType::kBatchValue);
  tensor::Tensor batch_mha_output = get_buffer(ModelBufferType::kBatchOutputMHA);
  tensor::Tensor batch_attn_output = get_buffer(ModelBufferType::kBatchAttnOutput);
  tensor::Tensor batch_ffn_rms = get_buffer(ModelBufferType::kBatchFFNRMSNorm);
  tensor::Tensor batch_w1_output = get_buffer(ModelBufferType::kBatchW1Output);
  tensor::Tensor batch_w3_output = get_buffer(ModelBufferType::kBatchW3Output);
  tensor::Tensor batch_w2_output = get_buffer(ModelBufferType::kBatchW2Output);

  // Get paged KV cache tensors
  const tensor::Tensor& kv_cache = kv_cache_manager_->kv_cache_device();
  const tensor::Tensor& block_tables = kv_cache_manager_->block_tables_device();
  const tensor::Tensor& context_lens = kv_cache_manager_->context_lens_device();

  // Get parameters
  const int32_t num_heads = config_->head_num_;
  const int32_t num_kv_heads = config_->kv_head_num_;
  const int32_t head_dim = config_->head_size_;
  const int32_t num_layers = config_->layer_num_;
  const int32_t block_size = kv_cache_manager_->block_size();
  const int32_t max_blocks_per_seq = kv_cache_manager_->max_blocks_per_seq();
  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;

  // Copy input to working buffer
  tensor::Tensor working_input = inputs;  // [num_seqs, dim]

  for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    // 1. Attention RMSNorm (batch) - use dim version for batch
    auto rmsnorm_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->rmsnorm_layers_.at(layer_idx));
    kernel::get_rmsnorm_dim_kernel(device_type_)(working_input, rmsnorm_layer->get_weight(0),
                                                  batch_rms_output, dim, stream);

    // 2. QKV projection (batch GEMM)
    // Q = rms_output @ Wq^T
    auto wq_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->wq_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(batch_rms_output, wq_layer->get_weight(0),
                                           batch_query, cuda_config_.get());

    // K = rms_output @ Wk^T
    auto wk_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->wk_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(batch_rms_output, wk_layer->get_weight(0),
                                           batch_key, cuda_config_.get());

    // V = rms_output @ Wv^T
    auto wv_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->wv_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(batch_rms_output, wv_layer->get_weight(0),
                                           batch_value, cuda_config_.get());

    // 3. RoPE (batch)
    kernel::get_rope_batch_kernel(device_type_)(
        num_seqs, dim, kv_dim, head_dim,
        batch_query, batch_key, positions,
        get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache),
        stream);

    // 4. Write K, V to paged KV cache (batch)
    kernel::batch_kv_cache_write_cu(
        const_cast<tensor::Tensor&>(kv_cache),
        batch_key,
        batch_value,
        block_tables,
        positions,
        num_seqs,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        layer_idx,
        num_layers,
        stream);

    // 5. Paged Attention (already supports batch)
    // Get max context len across all sequences
    int32_t max_context_len = 0;
    for (int32_t s = 0; s < num_seqs; ++s) {
      int32_t ctx_len = kv_cache_manager_->get_context_len(s);
      if (ctx_len > max_context_len) max_context_len = ctx_len;
    }

    kernel::paged_attention_kernel_cu(
        batch_mha_output,
        batch_query,
        kv_cache,
        block_tables,
        context_lens,
        max_context_len,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        scale,
        layer_idx,
        num_layers,
        stream);

    // 6. Output projection: O = mha_output @ Wo^T
    auto wo_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->wo_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(batch_mha_output, wo_layer->get_weight(0),
                                           batch_attn_output, cuda_config_.get());

    // 7. Residual add: working_input += attn_output
    kernel::get_add_kernel(device_type_)(working_input, batch_attn_output, working_input, stream);

    // 8. FFN RMSNorm
    auto ffn_rmsnorm = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->rmsnorm_layers_.at(layer_idx + num_layers));
    kernel::get_rmsnorm_dim_kernel(device_type_)(working_input, ffn_rmsnorm->get_weight(0),
                                                  batch_ffn_rms, dim, stream);

    // 9. FFN: W1 and W3
    auto w1_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->w1_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(batch_ffn_rms, w1_layer->get_weight(0),
                                           batch_w1_output, cuda_config_.get());

    auto w3_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->w3_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(batch_ffn_rms, w3_layer->get_weight(0),
                                           batch_w3_output, cuda_config_.get());

    // 10. SwiGLU
    kernel::get_swiglu_kernel(device_type_)(batch_w1_output, batch_w3_output, batch_w1_output, stream);

    // 11. W2
    auto w2_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->w2_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(batch_w1_output, w2_layer->get_weight(0),
                                           batch_w2_output, cuda_config_.get());

    // 12. Residual add
    kernel::get_add_kernel(device_type_)(working_input, batch_w2_output, working_input, stream);
  }

  // Final RMSNorm
  auto final_rmsnorm = std::dynamic_pointer_cast<op::LayerParam>(
      llama_layers_->rmsnorm_layers_.at(2 * num_layers));
  kernel::get_rmsnorm_dim_kernel(device_type_)(working_input, final_rmsnorm->get_weight(0),
                                                batch_rms_output, dim, stream);

  // Classification layer: logits = rms_output @ cls_weight^T
  // Output: [num_seqs, vocab_size]
  tensor::Tensor batch_forward_output = get_buffer(ModelBufferType::kBatchForwardOutput);
  auto cls_layer = std::dynamic_pointer_cast<op::LayerParam>(llama_layers_->cls_layer_);
  kernel::get_gemm_kernel(device_type_)(batch_rms_output, cls_layer->get_weight(0),
                                         batch_forward_output, cuda_config_.get());

  cudaStreamSynchronize(stream);

  // Sample next token for each sequence
  next_tokens.resize(num_seqs);
  const int32_t vocab_size = config_->vocab_size_;
  const float* logits_ptr = batch_forward_output.ptr<float>();

  for (int32_t s = 0; s < num_seqs; ++s) {
    const float* seq_logits = logits_ptr + s * vocab_size;
    next_tokens[s] = static_cast<int32_t>(sampler_->sample(seq_logits, vocab_size, stream));
  }

  return base::error::Success();
}

// ============================================================================
// Unified forward implementation
// Supports: single decode, batch decode, chunked prefill, mixed prefill+decode
// ============================================================================

void LLama2Model::ensure_unified_buffers(int32_t num_tokens) const {
  // Cast away const for buffer initialization (buffers are mutable cache)
  const_cast<LLama2Model*>(this)->init_unified_buffers(num_tokens);
}

base::Status LLama2Model::forward_unified(
    const tensor::Tensor& hidden_states,
    const base::BatchMetadata& batch_meta,
    std::vector<int32_t>& next_tokens) const {

  if (hidden_states.is_empty()) {
    return base::error::InvalidArgument("The hidden_states tensor is empty.");
  }

  CHECK(kv_cache_manager_ != nullptr) << "KV cache manager not initialized for unified forward";

  const int32_t num_tokens = batch_meta.num_tokens;
  const int32_t num_seqs = batch_meta.num_seqs;
  const bool has_prefill = batch_meta.has_prefill();
  const bool has_decode = batch_meta.has_decode();

  // Ensure we have buffers for this many tokens
  ensure_unified_buffers(num_tokens);

  cudaStream_t stream = cuda_config_ ? cuda_config_->stream : nullptr;

  // Get unified buffers
  tensor::Tensor unified_rms_output = get_buffer(ModelBufferType::kUnifiedRMSNormOutput);
  tensor::Tensor unified_query = get_buffer(ModelBufferType::kUnifiedQuery);
  tensor::Tensor unified_key = get_buffer(ModelBufferType::kUnifiedKey);
  tensor::Tensor unified_value = get_buffer(ModelBufferType::kUnifiedValue);
  tensor::Tensor unified_mha_output = get_buffer(ModelBufferType::kUnifiedMHAOutput);
  tensor::Tensor unified_attn_output = get_buffer(ModelBufferType::kUnifiedAttnOutput);
  tensor::Tensor unified_ffn_rms = get_buffer(ModelBufferType::kUnifiedFFNRMSNorm);
  tensor::Tensor unified_w1_output = get_buffer(ModelBufferType::kUnifiedW1Output);
  tensor::Tensor unified_w3_output = get_buffer(ModelBufferType::kUnifiedW3Output);
  tensor::Tensor unified_w2_output = get_buffer(ModelBufferType::kUnifiedW2Output);

  // Reshape buffers for current num_tokens
  unified_rms_output.reshape({num_tokens, config_->dim_});
  unified_query.reshape({num_tokens, config_->dim_});
  unified_key.reshape({num_tokens, config_->kv_dim_});
  unified_value.reshape({num_tokens, config_->kv_dim_});
  unified_mha_output.reshape({num_tokens, config_->dim_});
  unified_attn_output.reshape({num_tokens, config_->dim_});
  unified_ffn_rms.reshape({num_tokens, config_->dim_});
  unified_w1_output.reshape({num_tokens, config_->hidden_dim_});
  unified_w3_output.reshape({num_tokens, config_->hidden_dim_});
  unified_w2_output.reshape({num_tokens, config_->dim_});

  // Get paged KV cache tensors
  const tensor::Tensor& kv_cache = kv_cache_manager_->kv_cache_device();
  const tensor::Tensor& block_tables = kv_cache_manager_->block_tables_device();
  const tensor::Tensor& context_lens = kv_cache_manager_->context_lens_device();

  // Get parameters
  const int32_t num_heads = config_->head_num_;
  const int32_t num_kv_heads = config_->kv_head_num_;
  const int32_t head_dim = config_->head_size_;
  const int32_t num_layers = config_->layer_num_;
  const int32_t block_size = kv_cache_manager_->block_size();
  const int32_t max_blocks_per_seq = kv_cache_manager_->max_blocks_per_seq();
  const int32_t dim = config_->dim_;
  const int32_t kv_dim = config_->kv_dim_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Working tensor - copy of hidden_states that we modify in-place
  tensor::Tensor working_input = hidden_states;  // [num_tokens, dim]

  for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    // 1. Attention RMSNorm: [num_tokens, dim] -> [num_tokens, dim]
    auto rmsnorm_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->rmsnorm_layers_.at(layer_idx));
    kernel::get_rmsnorm_dim_kernel(device_type_)(
        working_input, rmsnorm_layer->get_weight(0), unified_rms_output, dim, stream);

    // 2. QKV Projection using GEMM
    auto wq_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->wq_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(
        unified_rms_output, wq_layer->get_weight(0), unified_query, cuda_config_.get());

    auto wk_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->wk_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(
        unified_rms_output, wk_layer->get_weight(0), unified_key, cuda_config_.get());

    auto wv_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->wv_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(
        unified_rms_output, wv_layer->get_weight(0), unified_value, cuda_config_.get());

    // 3. RoPE: Apply rotary position embeddings
    kernel::get_rope_batch_kernel(device_type_)(
        num_tokens, dim, kv_dim, head_dim,
        unified_query, unified_key, batch_meta.positions,
        get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache),
        stream);

    // 4. Write K, V to paged KV cache using v2 kernel with seq_ids
    kernel::batch_kv_cache_write_v2_cu(
        const_cast<tensor::Tensor&>(kv_cache),
        unified_key,
        unified_value,
        block_tables,
        batch_meta.positions,
        batch_meta.seq_ids,
        num_tokens,
        num_kv_heads,
        head_dim,
        block_size,
        max_blocks_per_seq,
        layer_idx,
        num_layers,
        stream);

    // 5. Attention: Choose kernel based on scenario
    int32_t max_context_len = batch_meta.max_context_len();

    if (has_prefill) {
      // Use prefill and decode mixed kernel 
      kernel::paged_attention_prefill_cu(
          unified_mha_output,
          unified_query,
          kv_cache,
          block_tables,
          context_lens,
          batch_meta.seq_ids,
          batch_meta.positions,
          num_tokens,
          num_seqs,
          num_heads,
          num_kv_heads,
          head_dim,
          block_size,
          max_blocks_per_seq,
          scale,
          layer_idx,
          num_layers,
          stream);
    } else {
      // Pure decode: use standard paged attention kernel
      kernel::paged_attention_kernel_cu(
          unified_mha_output,
          unified_query,
          kv_cache,
          block_tables,
          context_lens,
          max_context_len,
          num_seqs,
          num_heads,
          num_kv_heads,
          head_dim,
          block_size,
          max_blocks_per_seq,
          scale,
          layer_idx,
          num_layers,
          stream);
    }

    // 6. Output projection: O = mha_output @ Wo^T
    auto wo_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->wo_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(
        unified_mha_output, wo_layer->get_weight(0), unified_attn_output, cuda_config_.get());

    // 7. Residual add: working_input += attn_output
    kernel::get_add_kernel(device_type_)(working_input, unified_attn_output, working_input, stream);

    // 8. FFN RMSNorm
    auto ffn_rmsnorm = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->rmsnorm_layers_.at(layer_idx + num_layers));
    kernel::get_rmsnorm_dim_kernel(device_type_)(
        working_input, ffn_rmsnorm->get_weight(0), unified_ffn_rms, dim, stream);

    // 9. FFN: W1 and W3
    auto w1_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->w1_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(
        unified_ffn_rms, w1_layer->get_weight(0), unified_w1_output, cuda_config_.get());

    auto w3_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->w3_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(
        unified_ffn_rms, w3_layer->get_weight(0), unified_w3_output, cuda_config_.get());

    // 10. SwiGLU
    kernel::get_swiglu_kernel(device_type_)(unified_w1_output, unified_w3_output, unified_w1_output, stream);

    // 11. W2
    auto w2_layer = std::dynamic_pointer_cast<op::LayerParam>(
        llama_layers_->w2_layers_.at(layer_idx));
    kernel::get_gemm_kernel(device_type_)(
        unified_w1_output, w2_layer->get_weight(0), unified_w2_output, cuda_config_.get());

    // 12. Residual add
    kernel::get_add_kernel(device_type_)(working_input, unified_w2_output, working_input, stream);
  }

  // Final RMSNorm
  auto final_rmsnorm = std::dynamic_pointer_cast<op::LayerParam>(
      llama_layers_->rmsnorm_layers_.at(2 * num_layers));
  kernel::get_rmsnorm_dim_kernel(device_type_)(
      working_input, final_rmsnorm->get_weight(0), unified_rms_output, dim, stream);

  // Get last token indices for each sequence (for logits computation)
  std::vector<int32_t> last_token_indices = batch_meta.get_last_token_indices();

  // Compute logits only for the last token of each sequence
  // For efficiency, we extract last tokens and do a smaller GEMM
  next_tokens.resize(num_seqs);
  auto cls_layer = std::dynamic_pointer_cast<op::LayerParam>(llama_layers_->cls_layer_);
  const int32_t vocab_size = config_->vocab_size_;

  // Allocate temporary buffer for last token hidden states and logits
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  }

  tensor::Tensor last_hidden(base::DataType::kDataTypeFp32, num_seqs, dim, true, alloc);
  tensor::Tensor logits(base::DataType::kDataTypeFp32, num_seqs, vocab_size, true, alloc);

  // Copy last token hidden states
  for (int32_t s = 0; s < num_seqs; ++s) {
    int32_t last_idx = last_token_indices[s];
    cudaMemcpyAsync(
        last_hidden.ptr<float>(s * dim),
        unified_rms_output.ptr<float>(last_idx * dim),
        dim * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // Compute logits: [num_seqs, dim] @ [vocab_size, dim]^T -> [num_seqs, vocab_size]
  kernel::get_gemm_kernel(device_type_)(last_hidden, cls_layer->get_weight(0), logits, cuda_config_.get());

  cudaStreamSynchronize(stream);

  // Sample next token for each sequence
  const float* logits_ptr = logits.ptr<float>();
  for (int32_t s = 0; s < num_seqs; ++s) {
    const float* seq_logits = logits_ptr + s * vocab_size;
    next_tokens[s] = static_cast<int32_t>(sampler_->sample(seq_logits, vocab_size, stream));
  }

  return base::error::Success();
}

}  // namespace model