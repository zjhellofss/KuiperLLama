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
  model.init(base::DeviceType::kDeviceCUDA);
  std::string sentence = "This"; // prompts
  const auto& tokens = model.encode(sentence);
  TICK(A)
  const auto s = model.forward(tokens, 32);
  TOCK(A)
  LOG(INFO) << s;
  return 0;
}