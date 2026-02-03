#ifndef KUIPER_BASE_KV_CACHE_MANAGER_H
#define KUIPER_BASE_KV_CACHE_MANAGER_H

#include <cstdint>
#include <memory>
#include <vector>
#include "base/alloc.h"
#include "base/block_allocator.h"
#include "base/buffer.h"
#include "tensor/tensor.h"

namespace base {

class KVCacheManager {
public:
  base::Status init_paged_token_major(
      base::DeviceType device_type,
      std::shared_ptr<base::DeviceAllocator> alloc_dev,
      std::shared_ptr<base::DeviceAllocator> alloc_cpu,
      int32_t num_seqs,
      int32_t num_layers,
      int32_t num_kv_heads,
      int32_t head_dim,
      int32_t block_size,
      int32_t max_blocks,
      int32_t max_blocks_per_seq,
      void* stream = nullptr);

  std::pair<tensor::Tensor, tensor::Tensor> slice_kv(
      int32_t seq_idx, int32_t layer_idx, int32_t token_pos);

  const tensor::Tensor& kv_cache_device() const;
  const tensor::Tensor& block_tables_device();   // 确保已同步到 device
  const tensor::Tensor& context_lens_device();   // 确保已同步到 device

  const tensor::Tensor& block_tables_host() const;
  const tensor::Tensor& context_lens_host() const;

  void sync_tables_to_device(void* stream = nullptr); // host->device

  void reset_sequence(int32_t seq_idx); // 释放该 seq 的所有 blocks，block_tables=-1, context_len=0
  void reset_all();

  // Getters for configuration
  int32_t block_size() const;
  int32_t max_blocks_per_seq() const;
  int32_t kv_dim() const;
  int32_t num_layers() const;
  int32_t num_kv_heads() const;
  int32_t head_dim() const;
  int32_t num_seqs() const;
  int32_t max_blocks() const;

  // Get context length for a sequence
  int32_t get_context_len(int32_t seq_idx) const;

  // Set context length directly (useful after prefill)
  void set_context_len(int32_t seq_idx, int32_t context_len);

  // Allocate blocks for multiple tokens at once (for prefill)
  // Returns true if allocation succeeded
  bool allocate_blocks_for_tokens(int32_t seq_idx, int32_t num_tokens);

  // Sequence slot management (for Scheduler)
  // Allocate a free sequence slot, returns slot index or -1 if none available
  int32_t allocate_sequence_slot();

  // Release a sequence slot (frees blocks and resets state)
  void release_sequence_slot(int32_t seq_idx);

  // Check if there are free sequence slots
  bool has_free_slot() const;

  // Get number of free sequence slots
  int32_t num_free_slots() const;

  // Get number of free blocks (for memory pressure check)
  int32_t num_free_blocks() const;

private:
  float* key_ptr_(int32_t physical_block_id, int32_t layer_idx, int32_t token_in_block);
  float* value_ptr_(int32_t physical_block_id, int32_t layer_idx, int32_t token_in_block);

  int32_t ensure_physical_block_(int32_t seq_idx, int32_t logical_block);

  // Internal sync without locking (caller must hold mu_)
  void sync_tables_to_device_unlocked_(void* stream);

private:
  mutable std::mutex mu_;
  base::DeviceType device_type_{base::DeviceType::kDeviceUnknown};

  int32_t num_seqs_{0}, num_layers_{0}, num_kv_heads_{0}, head_dim_{0};
  int32_t kv_dim_{0};
  int32_t block_size_{0}, max_blocks_{0}, max_blocks_per_seq_{0};

  std::shared_ptr<base::DeviceAllocator> alloc_dev_;
  std::shared_ptr<base::DeviceAllocator> alloc_cpu_;

  tensor::Tensor kv_cache_dev_;          // float, device
  tensor::Tensor block_tables_host_;     // int32, cpu
  tensor::Tensor context_lens_host_;     // int32, cpu
  tensor::Tensor block_tables_dev_;      // int32, device mirror
  tensor::Tensor context_lens_dev_;      // int32, device mirror

  std::unique_ptr<base::BlockAllocator> block_alloc_;
  bool tables_dirty_{true};

  // Sequence slot management
  std::vector<int32_t> free_seq_slots_;  // Available sequence slots
};

} // namespace base
#endif // KUIPER_BASE_KV_CACHE_MANAGER_H
