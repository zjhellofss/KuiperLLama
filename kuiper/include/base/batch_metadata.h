#ifndef KUIPER_INCLUDE_BASE_BATCH_METADATA_H_
#define KUIPER_INCLUDE_BASE_BATCH_METADATA_H_

#include <memory>
#include <vector>
#include "base/alloc.h"
#include "tensor/tensor.h"

namespace base {

// Unified batch metadata for forward pass
// Supports: single decode, batch decode, chunked prefill, mixed prefill+decode
struct BatchMetadata {
  int32_t num_tokens = 0;  // Total number of tokens in this batch
  int32_t num_seqs = 0;    // Number of sequences

  // GPU tensors
  tensor::Tensor seq_ids;         // [num_tokens] - sequence index for each token
  tensor::Tensor positions;       // [num_tokens] - position for each token
  tensor::Tensor seq_start_locs;  // [num_seqs + 1] - start location of each sequence in tokens

  // CPU copies for host-side logic
  std::vector<int32_t> seq_ids_cpu;
  std::vector<int32_t> positions_cpu;
  std::vector<int32_t> seq_start_locs_cpu;
  std::vector<int32_t> seq_lens_cpu;    // Current context length for each sequence
  std::vector<bool> is_prefill_cpu;     // Whether each sequence is in prefill phase

  // Query methods
  bool has_prefill() const;
  bool has_decode() const;
  int32_t max_prefill_len() const;
  int32_t max_context_len() const;
  int32_t get_num_prefill_tokens() const;
  int32_t get_num_decode_tokens() const;

  // Get the last token index for each sequence (for logits computation)
  std::vector<int32_t> get_last_token_indices() const;

  // Factory methods
  static BatchMetadata create_decode(
      const std::vector<int32_t>& seq_ids,
      const std::vector<int32_t>& positions,
      const std::vector<int32_t>& context_lens,
      std::shared_ptr<DeviceAllocator> alloc_gpu,
      std::shared_ptr<DeviceAllocator> alloc_cpu,
      void* stream = nullptr);

  static BatchMetadata create_prefill(
      int32_t seq_id,
      int32_t start_pos,
      int32_t num_tokens,
      int32_t context_len,
      std::shared_ptr<DeviceAllocator> alloc_gpu,
      std::shared_ptr<DeviceAllocator> alloc_cpu,
      void* stream = nullptr);

  static BatchMetadata create_mixed(
      const std::vector<int32_t>& seq_ids,
      const std::vector<int32_t>& token_counts,  // Number of tokens per sequence
      const std::vector<int32_t>& start_positions,
      const std::vector<int32_t>& context_lens,
      const std::vector<bool>& is_prefill,
      std::shared_ptr<DeviceAllocator> alloc_gpu,
      std::shared_ptr<DeviceAllocator> alloc_cpu,
      void* stream = nullptr);

 private:
  void sync_to_device(std::shared_ptr<DeviceAllocator> alloc_gpu,
                      std::shared_ptr<DeviceAllocator> alloc_cpu,
                      void* stream);
};

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_BATCH_METADATA_H_