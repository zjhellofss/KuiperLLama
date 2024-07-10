#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama2.h"
int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint_path tokenizer_path ";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];
  model::LLama2Model model(tokenizer_path, checkpoint_path);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
  std::string sentence = "This";  // prompts
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  int32_t total_steps = 128;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  TICK(A)
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      next = model.forward(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.forward(input, pos_tensor, is_prompt, next);
    }
    if (next == model.get_eos()) {
      break;
    }
    std::string word;
    if (is_prompt) {
      next = tokens.at(pos + 1);
      word = model.decode(next);
    } else {
      word = model.decode(next);
    }
    printf("%s ", word.c_str());
    fflush(stdout);
    pos += 1;
  }
  printf("\n");
  TOCK(A)
  return 0;
}

// #include <base/tick.h>
// #include <glog/logging.h>
// #include "../source/op/kernels/kernels_interface.h"
// #include "base/alloc.h"
// #include "model/llama2.h"
// int main(int argc, char* argv[]) {
//   auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
//   auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
//
//   int32_t size = 2048;
//
//   tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
//   tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
//   tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
//
//   std::random_device rd;
//   std::mt19937 mt(rd());
//   std::uniform_real_distribution<float> dist(0.f, 1.f);
//   for (int i = 0; i < size; ++i) {
//     in_cpu.index<float>(i) = dist(mt);
//     wei_cpu.index<float>(i) = dist(mt);
//   }
//
//   tensor::Tensor in_cu = in_cpu.clone();
//   tensor::Tensor wei_cu = wei_cpu.clone();
//   tensor::Tensor out_cu = out_cpu.clone();
//   in_cu.to_cuda(nullptr);
//   wei_cu.to_cuda(nullptr);
//   out_cu.to_cuda(nullptr);
//   cudaStream_t stream;
//   cudaStreamCreate(&stream);
//   kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
//   out_cu.to_cpu();
//
//   kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);
//   return 0;
// }