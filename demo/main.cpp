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
  auto init_status = model.init(base::DeviceType::kDeviceCPU);
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