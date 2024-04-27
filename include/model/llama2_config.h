#ifndef LC_INCLUDE_MODEL_LLAMA_CONFIG_H_
#define LC_INCLUDE_MODEL_LLAMA_CONFIG_H_
struct LlamaModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;
};
#endif  // LC_INCLUDE_MODEL_LLAMA_CONFIG_H_
