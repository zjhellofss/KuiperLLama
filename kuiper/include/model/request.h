#ifndef KUIPER_INCLUDE_MODEL_REQUEST_H_
#define KUIPER_INCLUDE_MODEL_REQUEST_H_

#include <vector>
#include <cstdint>

namespace model {

// Request structure for managing inference requests with paged attention
struct Request {
  int32_t req_id;                      // Unique request identifier
  std::vector<int32_t> tokens;         // Token sequence (prompt + generated tokens)
  std::vector<int32_t> block_table;    // Physical block IDs allocated for this request
  int32_t prompt_len;                  // Length of the original prompt
  bool is_finished;                    // Whether generation is complete

  explicit Request(int32_t id, const std::vector<int32_t>& prompt_tokens)
      : req_id(id),
        tokens(prompt_tokens),
        prompt_len(static_cast<int32_t>(prompt_tokens.size())),
        is_finished(false) {}

  // Get current sequence length (prompt + generated tokens)
  int32_t get_current_seq_len() const {
    return static_cast<int32_t>(tokens.size());
  }

  // Get number of generated tokens (excluding prompt)
  int32_t get_generated_len() const {
    return get_current_seq_len() - prompt_len;
  }

  // Add a newly generated token
  void add_token(int32_t token) {
    tokens.push_back(token);
  }

  // Mark request as finished
  void finish() {
    is_finished = true;
  }
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_REQUEST_H_
