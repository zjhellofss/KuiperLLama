#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/llama2.h>
#include <tensor/tensor.h>
#include "base/buffer.h"

TEST(test_llama_model, cpu1) {
  using namespace base;
  std::shared_ptr<base::CPUDeviceAllocator> alloc =
      std::make_shared<base::CPUDeviceAllocator>();

  const char* checkpoint_path = "/home/fss/llama2/llama2_7b.bin";  // e.g. out/model.bin
  const char* tokenizer_path = "/home/fss/llama2/tokenizer.model";
  model::LLama2Model model(tokenizer_path, checkpoint_path);
  auto status = model.init(base::DeviceType::kDeviceCPU);
  if (status) {
    std::string sentence = "Hi";  // prompts
    const auto& tokens = model.encode(sentence);
    const auto s = model.forward(tokens, 1);
    const float* logits =
        model.get_buffer(model::ModelBufferType::kForwardOutput).ptr<float>();
    ASSERT_NEAR(logits[0], -12.7976265, 1e-3f);
    ASSERT_NEAR(logits[32], -9.97821331, 1e-3f);
    ASSERT_NEAR(logits[128], -12.8054199, 1e-3f);
    ASSERT_NEAR(logits[256], -12.7876959, 1e-3f);
    ASSERT_NEAR(logits[512], 4.75685883, 1e-3f);
    ASSERT_NEAR(logits[613], -3.83690214, 1e-3f);
    ASSERT_NEAR(logits[1011], -3.34461427, 1e-3f);
    ASSERT_NEAR(logits[1022], -7.45470142, 1e-3f);
    ASSERT_NEAR(logits[1023], -1.00463259, 1e-3f);
  }
}