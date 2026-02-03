/**
 * @author    jintang-coder
 *   - BlockAllocator: Physical memory block pool management
 *
 * =============================================================================
 * BlockAllocator 物理块分配器
 * =============================================================================
 *
 * 功能: 管理GPU显存中的物理块池，类似操作系统的页帧分配器
 *
 * 核心数据结构:
 *   class BlockAllocator {
 *       int total_blocks_;           // 总块数
 *       int block_size_;             // 每块可存储的token数
 *       std::vector<int> free_blocks_; // 空闲块ID栈
 *   };
 *
 * ---
 * 具体例子: 初始化与分配过程
 *
 * 假设配置:
 *   total_blocks = 100
 *   block_size = 16 (每块存16个token的KV)
 *
 * 初始化后:
 *   free_blocks_ = [99, 98, 97, ..., 2, 1, 0]  // 栈结构，后进先出
 *
 * ---
 * 分配场景1: Prefill阶段 - 一次分配多块
 *
 *   // 序列有50个token，需要 ceil(50/16) = 4 个blocks
 *   std::vector<int> blocks = allocator.allocate(4);
 *   // 返回 blocks = [0, 1, 2, 3]
 *   // free_blocks_ = [99, 98, ..., 5, 4]
 *
 * ---
 * 分配场景2: Decode阶段 - 单块分配
 *
 *   // 当前token位置超出已分配块，需要新块
 *   int new_block = allocator.allocate();
 *   // 返回 new_block = 4
 *   // free_blocks_ = [99, 98, ..., 5]
 *
 * ---
 * 释放场景: 序列完成后归还
 *
 *   allocator.free({0, 1, 2, 3, 4});
 *   // free_blocks_ = [99, 98, ..., 5, 0, 1, 2, 3, 4]
 *   // 注意: 归还顺序不影响，下次分配从栈顶取
 *
 * ---
 * 与上层组件的关系:
 *
 *   KVCacheManager
 *        │
 *        │ allocate_blocks_for_tokens(seq_idx, num_tokens)
 *        │   └─> 计算需要多少blocks
 *        │   └─> 调用 block_alloc_->allocate()
 *        ▼
 *   BlockAllocator
 *        │
 *        │ allocate() / free()
 *        ▼
 *   free_blocks_ 栈
 *
 * ---
 * 内存布局示意:
 *
 *   GPU显存 KV Cache Pool:
 *   ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
 *   │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │   ...   │
 *   │ (seq0)  │ (seq0)  │ (seq1)  │ (seq1)  │ (free)  │         │
 *   └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
 *
 *   每个Block内部:
 *   [num_layers][2(K/V)][block_size][kv_dim]
 *
 */

#ifndef KUIPER_BASE_BLOCK_ALLOCATOR_H
#define KUIPER_BASE_BLOCK_ALLOCATOR_H

#include <vector>
#include <mutex>
#include <algorithm>


namespace base {

// 显存物理逻辑块管理器
class BlockAllocator {
public:
    // 初始化：构建空闲块链表（栈）
    explicit BlockAllocator(int total_blocks, int block_size)
        : total_blocks_(total_blocks), block_size_(block_size) {
        free_blocks_.reserve(total_blocks);
        for (int i = 0; i < total_blocks; ++i) {
            // 初始时所有块都是空闲的
            free_blocks_.push_back(total_blocks - 1 - i); 
        }
    }

    // 分配 n 个块，成功返回 block_ids，失败返回空
    std::vector<int> allocate(int num_blocks) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.size() < num_blocks) {
            return {}; // OOM (Out of Memory)
        }
        std::vector<int> allocated_blocks;
        for (int i = 0; i < num_blocks; ++i) {
            allocated_blocks.push_back(free_blocks_.back());
            free_blocks_.pop_back();
        }
        return allocated_blocks;
    }

    // 分配单个块（用于 Decode 阶段）
    int allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_blocks_.empty()) {
            return -1; // OOM
        }
        int block_id = free_blocks_.back();
        free_blocks_.pop_back();
        return block_id;
    }

    // 释放块，归还到池中
    void free(const std::vector<int>& blocks) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int block_id : blocks) {
            free_blocks_.push_back(block_id);
        }
    }

    int get_free_block_num() const {
        return free_blocks_.size();
    }

    int block_size() const { return block_size_; }

private:
    int total_blocks_;
    int block_size_;
    std::vector<int> free_blocks_; // 空闲块 ID 栈
    std::mutex mutex_;
};

} // namespace base

#endif // KUIPER_BASE_BLOCK_ALLOCATOR_H