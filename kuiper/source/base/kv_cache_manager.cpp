/**
 * @author    jintang-coder
 *   - KVCacheManager: KV Cache memory and sequence slot management
 *
 * =============================================================================
 * KVCacheManager KV缓存管理器
 * =============================================================================
 *
 * 功能: 管理序列槽位(slot)分配 + 物理块(block)映射 + KV Cache读写
 *
 * 核心数据结构:
 *   class KVCacheManager {
 *       tensor::Tensor kv_cache_dev_;       // GPU上的KV缓存池
 *       tensor::Tensor block_tables_host_;  // [num_seqs, max_blocks_per_seq] 逻辑块→物理块映射
 *       tensor::Tensor context_lens_host_;  // [num_seqs] 每个序列的当前长度
 *       std::vector<int32_t> free_seq_slots_; // 空闲序列槽位
 *       BlockAllocator block_alloc_;        // 物理块分配器
 *   };
 *
 * ---
 * 具体例子: 3个序列的完整生命周期
 *
 * 配置:
 *   num_seqs = 4, block_size = 16, max_blocks = 100
 *
 * 初始状态:
 *   free_seq_slots_ = [3, 2, 1, 0]
 *   block_tables_ = [[-1,-1,...], [-1,-1,...], [-1,-1,...], [-1,-1,...]]
 *   context_lens_ = [0, 0, 0, 0]
 *
 * ---
 * 步骤1: 分配序列槽位
 *
 *   int slot = kv_manager->allocate_sequence_slot();
 *   // slot = 0
 *   // free_seq_slots_ = [3, 2, 1]
 *
 * ---
 * 步骤2: 为Prefill分配blocks (假设prompt有50个tokens)
 *
 *   kv_manager->allocate_blocks_for_tokens(slot=0, num_tokens=50);
 *   // 需要 ceil(50/16) = 4 个blocks
 *   // 调用 block_alloc_->allocate() 4次，获得 blocks [0,1,2,3]
 *
 *   block_tables_[0] = [0, 1, 2, 3, -1, -1, ...]
 *   //                  │  │  │  │
 *   //                  │  │  │  └─ logical_block_3 → physical_block_3
 *   //                  │  │  └─ logical_block_2 → physical_block_2
 *   //                  │  └─ logical_block_1 → physical_block_1
 *   //                  └─ logical_block_0 → physical_block_0
 *
 *   context_lens_[0] = 50
 *
 * ---
 * 步骤3: Decode阶段 - 生成新token
 *
 *   // token位置从50增长到64，仍在block_3内 (16*4=64)
 *   // 无需新块
 *
 *   // token位置增长到65，需要新块
 *   kv_manager->allocate_blocks_for_tokens(slot=0, num_tokens=65);
 *   // logical_block_4 需要分配 → physical_block_4
 *   block_tables_[0] = [0, 1, 2, 3, 4, -1, ...]
 *
 * ---
 * 步骤4: 序列完成 - 释放资源
 *
 *   kv_manager->release_sequence_slot(slot=0);
 *   // 1. 收集 blocks [0,1,2,3,4]
 *   // 2. 调用 block_alloc_->free([0,1,2,3,4])
 *   // 3. block_tables_[0] = [-1,-1,-1,...]
 *   // 4. context_lens_[0] = 0
 *   // 5. free_seq_slots_.push_back(0) → [3,2,1,0]
 *
 * ---
 * KV Cache 内存布局:
 *
 *   kv_cache_dev_ shape: [max_blocks, num_layers, 2, block_size, kv_dim]
 *
 *   访问 seq0, layer2, token位置35 的 Key:
 *     logical_block = 35 / 16 = 2
 *     token_in_block = 35 % 16 = 3
 *     physical_block = block_tables_[0][2] = 2
 *
 *     offset = physical_block * (num_layers * 2 * block_size * kv_dim)
 *            + layer_idx * (2 * block_size * kv_dim)
 *            + 0 * (block_size * kv_dim)  // K plane
 *            + token_in_block * kv_dim
 *
 * ---
 * 与Scheduler的交互:
 *
 *   Scheduler                         KVCacheManager
 *      │                                    │
 *      │ admit_sequence()                   │
 *      │   └─> allocate_sequence_slot() ───►│ 返回 slot
 *      │   └─> allocate_blocks_for_tokens()►│ 分配物理块
 *      │                                    │
 *      │ finish_sequence()                  │
 *      │   └─> release_sequence_slot() ────►│ 释放slot+blocks
 *      │                                    │
 *
 */

#include "base/kv_cache_manager.h"
#include <glog/logging.h>
#include <cuda_runtime.h>
#include "tensor/tensor.h"

namespace base {

base::Status KVCacheManager::init_paged_token_major(
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
    void* stream) {

  std::lock_guard<std::mutex> lock(mu_);

  device_type_ = device_type;
  num_seqs_ = num_seqs;
  num_layers_ = num_layers;
  num_kv_heads_ = num_kv_heads;
  head_dim_ = head_dim;
  kv_dim_ = num_kv_heads * head_dim;
  block_size_ = block_size;
  max_blocks_ = max_blocks;
  max_blocks_per_seq_ = max_blocks_per_seq;

  alloc_dev_ = alloc_dev;
  alloc_cpu_ = alloc_cpu;

  // Initialize block allocator
  block_alloc_ = std::make_unique<BlockAllocator> (max_blocks, block_size);

  // Allocate KV cache on device
  // Layout: [max_blocks, num_layers, 2 (K+V), block_size, kv_dim]
  size_t kv_cache_size = static_cast<size_t>(max_blocks) * num_layers * 2 * block_size * kv_dim_;
  kv_cache_dev_ = tensor::Tensor(base::DataType::kDataTypeFp32, kv_cache_size);
  kv_cache_dev_.allocate(alloc_dev_);
  kv_cache_dev_.set_device_type(device_type);

  // Initialize to zero
  if (device_type == base::DeviceType::kDeviceCUDA) {
    cudaMemsetAsync(kv_cache_dev_.ptr<void>(), 0, kv_cache_size * sizeof(float),
                    static_cast<cudaStream_t>(stream));
  } else {
    memset(kv_cache_dev_.ptr<void>(), 0, kv_cache_size * sizeof(float));
  }

  // Allocate block tables on host and device
  // Layout: [num_seqs, max_blocks_per_seq]
  int32_t block_table_size = num_seqs * max_blocks_per_seq;
  block_tables_host_ = tensor::Tensor(base::DataType::kDataTypeInt32, block_table_size);
  block_tables_host_.allocate(alloc_cpu_);
  block_tables_host_.set_device_type(base::DeviceType::kDeviceCPU);

  block_tables_dev_ = tensor::Tensor(base::DataType::kDataTypeInt32, block_table_size);
  block_tables_dev_.allocate(alloc_dev_);
  block_tables_dev_.set_device_type(device_type);

  // Initialize block tables to -1 (invalid)
  int32_t* block_table_ptr = block_tables_host_.ptr<int32_t>();
  for (int32_t i = 0; i < block_table_size; ++i) {
    block_table_ptr[i] = -1;
  }

  // Allocate context lengths on host and device
  // Layout: [num_seqs]
  context_lens_host_ = tensor::Tensor(base::DataType::kDataTypeInt32, num_seqs);
  context_lens_host_.allocate(alloc_cpu_);
  context_lens_host_.set_device_type(base::DeviceType::kDeviceCPU);

  context_lens_dev_ = tensor::Tensor(base::DataType::kDataTypeInt32, num_seqs);
  context_lens_dev_.allocate(alloc_dev_);
  context_lens_dev_.set_device_type(device_type);
  // Initialize context lengths to 0
  int32_t* context_lens_ptr = context_lens_host_.ptr<int32_t>();
  for (int32_t i = 0; i < num_seqs; ++i) {
    context_lens_ptr[i] = 0;
  }

  // Initialize free sequence slots (reverse order so we allocate 0, 1, 2, ...)
  free_seq_slots_.clear();
  free_seq_slots_.reserve(num_seqs);
  for (int32_t i = num_seqs - 1; i >= 0; --i) {
    free_seq_slots_.push_back(i);
  }

  tables_dirty_ = true;

  return base::error::Success();
}

std::pair<tensor::Tensor, tensor::Tensor> KVCacheManager::slice_kv(
    int32_t seq_idx, int32_t layer_idx, int32_t token_pos) {

  std::lock_guard<std::mutex> lock(mu_);

  CHECK_LT(seq_idx, num_seqs_) << "Sequence index out of bounds";
  CHECK_LT(layer_idx, num_layers_) << "Layer index out of bounds";

  // Determine which logical block this token belongs to
  int32_t logical_block = token_pos / block_size_;
  int32_t token_in_block = token_pos % block_size_;

  // Get or allocate physical block
  int32_t physical_block_id = ensure_physical_block_(seq_idx, logical_block);
  CHECK_GE(physical_block_id, 0) << "Failed to allocate physical block";

  // Update context length if needed
  int32_t* context_lens_ptr = context_lens_host_.ptr<int32_t>();
  if (token_pos >= context_lens_ptr[seq_idx]) {
    context_lens_ptr[seq_idx] = token_pos + 1;
    tables_dirty_ = true;
  }

  // Calculate pointers to K and V
 float* key_ptr = key_ptr_(physical_block_id, layer_idx, token_in_block);
 float* value_ptr = value_ptr_(physical_block_id, layer_idx, token_in_block);

  // Create tensor views (non-owning)
  tensor::Tensor key(base::DataType::kDataTypeFp32, kv_dim_, false, nullptr, key_ptr);
  tensor::Tensor val(base::DataType::kDataTypeFp32, kv_dim_, false, nullptr, value_ptr);

  key.set_device_type(device_type_);
  val.set_device_type(device_type_);

  return {key, val};
}

const tensor::Tensor& KVCacheManager::kv_cache_device() const {
  return kv_cache_dev_;
}

const tensor::Tensor& KVCacheManager::block_tables_device() {
  std::lock_guard<std::mutex> lock(mu_);
  if (tables_dirty_) {
    sync_tables_to_device_unlocked_(nullptr);
  }
  return block_tables_dev_;
}

const tensor::Tensor& KVCacheManager::context_lens_device() {
  std::lock_guard<std::mutex> lock(mu_);
  if (tables_dirty_) {
    sync_tables_to_device_unlocked_(nullptr);
  }
  return context_lens_dev_;
}

const tensor::Tensor& KVCacheManager::block_tables_host() const {
  return block_tables_host_;
}

const tensor::Tensor& KVCacheManager::context_lens_host() const {
  return context_lens_host_;
}

void KVCacheManager::sync_tables_to_device(void* stream) {
  std::lock_guard<std::mutex> lock(mu_);
  sync_tables_to_device_unlocked_(stream);
}

void KVCacheManager::sync_tables_to_device_unlocked_(void* stream) {
  if (!tables_dirty_) {
    return;
  }


  // Copy block tables
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    cudaMemcpyAsync(block_tables_dev_.ptr<void>(),
                    block_tables_host_.ptr<void>(),
                    block_tables_host_.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));

    cudaMemcpyAsync(context_lens_dev_.ptr<void>(),
                    context_lens_host_.ptr<void>(),
                    context_lens_host_.size() * sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream));
  } else {
    memcpy(block_tables_dev_.ptr<void>(),
           block_tables_host_.ptr<void>(),
           block_tables_host_.size() * sizeof(int32_t));

    memcpy(context_lens_dev_.ptr<void>(),
           context_lens_host_.ptr<void>(),
           context_lens_host_.size() * sizeof(int32_t));
  }

  tables_dirty_ = false;
}

void KVCacheManager::reset_sequence(int32_t seq_idx) {
  std::lock_guard<std::mutex> lock(mu_);

  CHECK_LT(seq_idx, num_seqs_) << "Sequence index out of bounds";

  // Free all blocks for this sequence
  std::vector<int32_t> blocks_to_free;
  int32_t* block_table_ptr = block_tables_host_.ptr<int32_t>();

  for (int32_t i = 0; i < max_blocks_per_seq_; ++i) {
    int32_t block_id = block_table_ptr[seq_idx * max_blocks_per_seq_ + i];
    if (block_id >= 0) {
      blocks_to_free.push_back(block_id);
      block_table_ptr[seq_idx * max_blocks_per_seq_ + i] = -1;
    }
  }

  if (!blocks_to_free.empty()) {
    block_alloc_->free(blocks_to_free);
  }

  // Reset context length
  int32_t* context_lens_ptr = context_lens_host_.ptr<int32_t>();
  context_lens_ptr[seq_idx] = 0;

  tables_dirty_ = true;
}

void KVCacheManager::reset_all() {
  std::lock_guard<std::mutex> lock(mu_);

  // Reset all sequences
  int32_t* block_table_ptr = block_tables_host_.ptr<int32_t>();
  for (int32_t i = 0; i < num_seqs_ * max_blocks_per_seq_; ++i) {
    block_table_ptr[i] = -1;
  }

  int32_t* context_lens_ptr = context_lens_host_.ptr<int32_t>();
  for (int32_t i = 0; i < num_seqs_; ++i) {
    context_lens_ptr[i] = 0;
  }

  // Reinitialize block allocator
  block_alloc_ = std::make_unique<BlockAllocator>(max_blocks_, block_size_);

  tables_dirty_ = true;
}

int32_t KVCacheManager::block_size() const {
  return block_size_;
}

int32_t KVCacheManager::max_blocks_per_seq() const {
  return max_blocks_per_seq_;
}

int32_t KVCacheManager::kv_dim() const {
  return kv_dim_;
}

int32_t KVCacheManager::num_layers() const {
  return num_layers_;
}

int32_t KVCacheManager::num_kv_heads() const {
  return num_kv_heads_;
}

int32_t KVCacheManager::head_dim() const {
  return head_dim_;
}

int32_t KVCacheManager::num_seqs() const {
  return num_seqs_;
}

int32_t KVCacheManager::max_blocks() const {
  return max_blocks_;
}

int32_t KVCacheManager::get_context_len(int32_t seq_idx) const {
  std::lock_guard<std::mutex> lock(mu_);
  CHECK_LT(seq_idx, num_seqs_) << "Sequence index out of bounds";
  return context_lens_host_.ptr<int32_t>()[seq_idx];
}

void KVCacheManager::set_context_len(int32_t seq_idx, int32_t context_len) {
  std::lock_guard<std::mutex> lock(mu_);
  CHECK_LT(seq_idx, num_seqs_) << "Sequence index out of bounds";
  CHECK_GE(context_len, 0) << "Context length must be non-negative";

  int32_t* context_lens_ptr = context_lens_host_.ptr<int32_t>();
  context_lens_ptr[seq_idx] = context_len;
  tables_dirty_ = true;
}

bool KVCacheManager::allocate_blocks_for_tokens(int32_t seq_idx, int32_t num_tokens) {
  std::lock_guard<std::mutex> lock(mu_);
  CHECK_LT(seq_idx, num_seqs_) << "Sequence index out of bounds";

  if (num_tokens <= 0) {
    return true;
  }

  // Calculate how many blocks we need
  int32_t num_blocks_needed = (num_tokens + block_size_ - 1) / block_size_;

  if (num_blocks_needed > max_blocks_per_seq_) {
    LOG(ERROR) << "Requested " << num_blocks_needed << " blocks, but max_blocks_per_seq is "
               << max_blocks_per_seq_;
    return false;
  }

  // Check current allocation and allocate missing blocks
  int32_t* block_table_ptr = block_tables_host_.ptr<int32_t>();

  for (int32_t logical_block = 0; logical_block < num_blocks_needed; ++logical_block) {
    int32_t table_idx = seq_idx * max_blocks_per_seq_ + logical_block;
    if (block_table_ptr[table_idx] < 0) {
      // Need to allocate this block
      int32_t physical_block_id = block_alloc_->allocate();
      if (physical_block_id < 0) {
        LOG(ERROR) << "Out of memory: cannot allocate block " << logical_block;
        return false;
      }
      block_table_ptr[table_idx] = physical_block_id;
      tables_dirty_ = true;
    }
  }

  // Update context length
  int32_t* context_lens_ptr = context_lens_host_.ptr<int32_t>();
  if (num_tokens > context_lens_ptr[seq_idx]) {
    context_lens_ptr[seq_idx] = num_tokens;
    tables_dirty_ = true;
  }

  return true;
}


// Private helper methods

 float* KVCacheManager::key_ptr_(int32_t physical_block_id, int32_t layer_idx,
                                int32_t token_in_block)  {
  // Layout: [max_blocks, num_layers, 2 (K+V), block_size, kv_dim]
  size_t offset = static_cast<size_t>(physical_block_id) * num_layers_ * 2 * block_size_ * kv_dim_
                + static_cast<size_t>(layer_idx) * 2 * block_size_ * kv_dim_
                + 0 * block_size_ * kv_dim_  // K plane (index 0)
                + static_cast<size_t>(token_in_block) * kv_dim_;

  return (kv_cache_dev_.ptr<float>()) + offset;
}

 float* KVCacheManager::value_ptr_(int32_t physical_block_id, int32_t layer_idx,
                                  int32_t token_in_block)  {
  // Layout: [max_blocks, num_layers, 2 (K+V), block_size, kv_dim]
  size_t offset = static_cast<size_t>(physical_block_id) * num_layers_ * 2 * block_size_ * kv_dim_
                + static_cast<size_t>(layer_idx) * 2 * block_size_ * kv_dim_
                + 1 * block_size_ * kv_dim_  // V plane (index 1)
                + static_cast<size_t>(token_in_block) * kv_dim_;

  return  (kv_cache_dev_.ptr<float>()) + offset;
}

int32_t KVCacheManager::ensure_physical_block_(int32_t seq_idx, int32_t logical_block) {
  CHECK_LT(logical_block, max_blocks_per_seq_) << "Logical block index out of bounds";

  int32_t* block_table_ptr = block_tables_host_.ptr<int32_t>();
  int32_t table_idx = seq_idx * max_blocks_per_seq_ + logical_block;

  int32_t physical_block_id = block_table_ptr[table_idx];

  // If not allocated, allocate a new block
  if (physical_block_id < 0) {
    physical_block_id = block_alloc_->allocate();
    CHECK_GE(physical_block_id, 0) << "Out of memory: cannot allocate new block";

    block_table_ptr[table_idx] = physical_block_id;
    tables_dirty_ = true;
  }

  return physical_block_id;
}

int32_t KVCacheManager::allocate_sequence_slot() {
  std::lock_guard<std::mutex> lock(mu_);
  if (free_seq_slots_.empty()) {
    return -1;
  }
  int32_t slot = free_seq_slots_.back();
  free_seq_slots_.pop_back();
  return slot;
}

void KVCacheManager::release_sequence_slot(int32_t seq_idx) {
  std::lock_guard<std::mutex> lock(mu_);
  if (seq_idx < 0 || seq_idx >= num_seqs_) {
    return;
  }

  // Free all blocks for this sequence
  std::vector<int32_t> blocks_to_free;
  int32_t* block_table_ptr = block_tables_host_.ptr<int32_t>();

  for (int32_t i = 0; i < max_blocks_per_seq_; ++i) {
    int32_t block_id = block_table_ptr[seq_idx * max_blocks_per_seq_ + i];
    if (block_id >= 0) {
      blocks_to_free.push_back(block_id);
      block_table_ptr[seq_idx * max_blocks_per_seq_ + i] = -1;
    }
  }

  if (!blocks_to_free.empty()) {
    block_alloc_->free(blocks_to_free);
  }

  // Reset context length
  int32_t* context_lens_ptr = context_lens_host_.ptr<int32_t>();
  context_lens_ptr[seq_idx] = 0;

  tables_dirty_ = true;

  // Return slot to free pool
  free_seq_slots_.push_back(seq_idx);
}

bool KVCacheManager::has_free_slot() const {
  std::lock_guard<std::mutex> lock(mu_);
  return !free_seq_slots_.empty();
}

int32_t KVCacheManager::num_free_slots() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int32_t>(free_seq_slots_.size());
}

int32_t KVCacheManager::num_free_blocks() const {
  std::lock_guard<std::mutex> lock(mu_);
  return block_alloc_->get_free_block_num();
}

}  // namespace base
