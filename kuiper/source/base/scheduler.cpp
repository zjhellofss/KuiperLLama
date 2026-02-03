/**
 * @author    jintang-coder
 *   - Scheduler: Request scheduling and continuous batching
 *
 * =============================================================================
 * Scheduler 调度器
 * =============================================================================
 *
 * 功能: 管理推理请求队列，决定每次迭代处理哪些序列的哪些tokens
 *
 * 核心数据结构:
 *   class Scheduler {
 *       std::deque<int32_t> waiting_queue_;   // 等待队列 (新请求)
 *       std::deque<int32_t> running_queue_;   // 运行队列 (已分配KV slot)
 *       std::vector<int32_t> finished_seqs_;  // 已完成序列
 *       std::unordered_map<int32_t, Sequence> sequences_; // 序列详情
 *       KVCacheManager* kv_manager_;          // KV缓存管理器
 *   };
 *
 * ---
 * 具体例子: 3个请求的调度过程
 *
 * 请求到达:
 *   Seq0: "Hello world" (10 tokens), max_new_tokens=50
 *   Seq1: "Once upon a time" (20 tokens), max_new_tokens=50
 *   Seq2: "The quick brown fox" (15 tokens), max_new_tokens=50
 *
 * 配置:
 *   max_num_batched_tokens = 32
 *   max_prefill_tokens = 16 (chunked prefill)
 *   decode_priority = true
 *
 * ---
 * 迭代1: schedule() - 首次调度
 *
 *   waiting_queue_ = [0, 1, 2]
 *   running_queue_ = []
 *
 *   1. schedule_decode() - 无decode序列，跳过
 *   2. schedule_prefill() - 无running序列，跳过
 *   3. try_admit_waiting() - 从waiting移入running
 *
 *   remaining_tokens = 32
 *
 *   admit_sequence(seq0):
 *     kv_slot = kv_manager_->allocate_sequence_slot() → 0
 *     chunk_size = min(32, 16, 10) = 10
 *     remaining_tokens = 32 - 10 = 22
 *
 *   admit_sequence(seq1):
 *     kv_slot = kv_manager_->allocate_sequence_slot() → 1
 *     chunk_size = min(22, 16, 20) = 16
 *     remaining_tokens = 22 - 16 = 6
 *
 *   admit_sequence(seq2):
 *     kv_slot = kv_manager_->allocate_sequence_slot() → 2
 *     chunk_size = min(6, 16, 15) = 6  // 用完剩余token额度
 *     remaining_tokens = 6 - 6 = 0
 *
 *   输出 SchedulerOutput:
 *     scheduled_seqs = [
 *       {seq_id=0, kv_slot=0, num_tokens=10, start_pos=0, is_prefill=true},
 *       {seq_id=1, kv_slot=1, num_tokens=16, start_pos=0, is_prefill=true},
 *       {seq_id=2, kv_slot=2, num_tokens=6,  start_pos=0, is_prefill=true},
 *     ]
 *     num_tokens = 32, num_prefill_tokens = 32
 *
 *   状态变化:
 *     waiting_queue_ = []  // 全部admit
 *     running_queue_ = [0, 1, 2]
 *     seq0.prefill_pos = 0 → 10 (prefill完成)
 *     seq1.prefill_pos = 0 → 16 (部分prefill, 还剩4个)
 *     seq2.prefill_pos = 0 → 6  (部分prefill, 还剩9个)
 *
 * ---
 * 迭代2: schedule() - 混合批次 (Prefill续 + Decode)
 *
 *   running_queue_ = [0, 1, 2]
 *   remaining_tokens = 32
 *
 *   1. schedule_decode() - seq0已完成prefill，加入decode
 *      {seq_id=0, num_tokens=1, start_pos=10, is_prefill=false}
 *      remaining_tokens = 32 - 1 = 31
 *
 *   2. schedule_prefill() - seq1还有4个token, seq2还有9个token
 *      seq1: chunk_size = min(31, 16, 4) = 4
 *      {seq_id=1, num_tokens=4, start_pos=16, is_prefill=true}
 *      remaining_tokens = 31 - 4 = 27
 *
 *      seq2: chunk_size = min(27, 16, 9) = 9
 *      {seq_id=2, num_tokens=9, start_pos=6, is_prefill=true}
 *      remaining_tokens = 27 - 9 = 18
 *
 *   3. try_admit_waiting() - waiting为空，跳过
 *
 *   输出 SchedulerOutput:
 *     scheduled_seqs = [
 *       {seq_id=0, kv_slot=0, num_tokens=1, is_prefill=false}, // decode
 *       {seq_id=1, kv_slot=1, num_tokens=4, is_prefill=true},  // prefill完成
 *       {seq_id=2, kv_slot=2, num_tokens=9, is_prefill=true},  // prefill完成
 *     ]
 *     num_tokens = 14
 *     num_decode_tokens = 1, num_prefill_tokens = 13
 *
 *   状态变化:
 *     seq0.prefill_pos = 10 (已完成), 进入decode
 *     seq1.prefill_pos = 16 → 20 (prefill完成)
 *     seq2.prefill_pos = 6 → 15 (prefill完成)
 *
 * ---
 * 迭代N: 全部进入decode阶段
 *
 *   所有序列prefill完成，每次迭代:
 *     scheduled_seqs = [
 *       {seq_id=0, num_tokens=1, is_prefill=false},
 *       {seq_id=1, num_tokens=1, is_prefill=false},
 *       {seq_id=2, num_tokens=1, is_prefill=false},
 *     ]
 *     num_tokens = 3, num_decode_tokens = 3
 *
 * ---
 * 序列完成: finish_sequence()
 *
 *   // 检测到EOS或达到max_new_tokens
 *   scheduler.finish_sequence(seq_id=0);
 *
 *   1. seq0.status = FINISHED
 *   2. running_queue_.erase(0) → [1, 2]
 *   3. kv_manager_->release_sequence_slot(kv_slot=0)
 *   4. finished_seqs_.push_back(0)
 *
 * ---
 * 调度策略 (decode_priority=true):
 *
 *   优先级: Decode > Prefill续 > 新Prefill
 *
 *   原因: Decode只需1个token，延迟敏感
 *         Prefill可以分chunk，吞吐优先
 *
 *   schedule() 执行顺序:
 *     1. schedule_decode()      - 先处理所有decode序列
 *     2. schedule_prefill()     - 再处理running中的prefill
 *     3. try_admit_waiting()    - 最后admit新序列
 *
 * ---
 * 与其他组件的关系:
 *
 *   用户请求
 *      │
 *      ▼
 *   Scheduler.add_request()
 *      │
 *      ▼
 *   Scheduler.schedule() ──────► SchedulerOutput
 *      │                              │
 *      │ admit_sequence()             │ scheduled_seqs
 *      ▼                              ▼
 *   KVCacheManager              BatchMetadata.create_mixed()
 *      │                              │
 *      │ allocate_sequence_slot()     │ 构建GPU映射数组
 *      │ allocate_blocks_for_tokens() │
 *      ▼                              ▼
 *   BlockAllocator              GPU Kernels
 *
 * 
 *  最终驱逐策略                                                                                                                                                    
                                                                                                                                                                  
  优先级 (从高到低驱逐):                                                                                                                                          
  ┌─────────────────────────────────────────────────────────────┐                                                                                                 
  │ 1. Prefill 阶段序列 - 按 context_len 从小到大               │                                                                                                 
  │    (还没产出，用户影响最小)                                  │                                                                                                
  ├─────────────────────────────────────────────────────────────┤                                                                                                 
  │ 2. Decode 阶段序列 - 按 num_generated 从小到大              │                                                                                                 
  │    (已在生成，用户正在等待)                                  │                                                                                                
  └─────────────────────────────────────────────────────────────┘                                                                                                 
                                                                                                                                                                  
  示例                                                                                                                                                            
                                                                                                                                                                  
  running_queue_ = [seq0, seq1, seq2, seq3]                                                                                                                       
                                                                                                                                                                  
  seq0: prefill, context_len=100  (prefill了100个token)                                                                                                           
  seq1: prefill, context_len=20   ← 第1个驱逐目标 (prefill最短)                                                                                                   
  seq2: decode,  num_generated=50 (已生成50个)                                                                                                                    
  seq3: decode,  num_generated=5  ← 第2个驱逐目标 (decode最短)                                                                                                    
                                                                                                                                                                  
  驱逐顺序:                                                                                                                                                       
  1. seq1 (prefill, context=20)   - 优先驱逐prefill中最短的                                                                                                       
  2. seq0 (prefill, context=100)  - 其次prefill中第二短的                                                                                                         
  3. seq3 (decode, generated=5)   - 再次decode中最短的                                                                                                            
  4. seq2 (decode, generated=50)  - 最后decode中第二短的  
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */

#include "base/scheduler.h"
#include "base/kv_cache_manager.h"
#include <algorithm>
#include <glog/logging.h>

namespace base {

Scheduler::Scheduler(const SchedulerConfig& config) : config_(config) {}

void Scheduler::set_kv_cache_manager(KVCacheManager* kv_manager) {
    std::lock_guard<std::mutex> lock(mutex_);
    kv_manager_ = kv_manager;
}

int32_t Scheduler::add_request(const std::vector<int32_t>& prompt_tokens,
                                int32_t max_new_tokens,
                                const std::string& original_text) {
    std::lock_guard<std::mutex> lock(mutex_);

    Sequence seq;
    seq.seq_id = next_seq_id_++;
    seq.prompt_tokens = prompt_tokens;
    seq.max_new_tokens = max_new_tokens;
    seq.original_text = original_text;
    seq.status = SequenceStatus::WAITING;
    seq.arrival_time = std::chrono::steady_clock::now();

    int32_t seq_id = seq.seq_id;
    sequences_[seq_id] = std::move(seq);
    waiting_queue_.push_back(seq_id);

    return seq_id;
}

SchedulerOutput Scheduler::schedule() {
    std::lock_guard<std::mutex> lock(mutex_);

    SchedulerOutput output;
    int32_t remaining_tokens = config_.max_num_batched_tokens;
    int32_t remaining_seqs = config_.max_num_seqs;

    // Step 0: Check memory pressure and preempt if necessary
    // Estimate blocks needed for decode (1 token per running decode sequence)
    int32_t decode_blocks_needed = 0;
    for (int32_t seq_id : running_queue_) {
        auto it = sequences_.find(seq_id);
        if (it != sequences_.end() && !it->second.is_prefill()) {
            // Each decode may need a new block if crossing block boundary
            decode_blocks_needed += estimate_blocks_needed(it->second, 1) -
                                    estimate_blocks_needed(it->second, 0);
        }
    }

    if (need_preemption(decode_blocks_needed)) {
        int32_t preempted = preempt_sequences(decode_blocks_needed);
        if (preempted > 0) {
            for (int32_t i = 0; i < preempted; ++i) {
                if (!preempted_queue_.empty()) {
                    output.preempted_seq_ids.push_back(preempted_queue_.back());
                }
            }
        }
    }

    if (config_.decode_priority) {
        // Step 1: Schedule decode sequences first (each needs only 1 token)
        schedule_decode(output, remaining_tokens, remaining_seqs);

        // Step 2: Schedule running prefill sequences
        schedule_prefill(output, remaining_tokens, remaining_seqs);

        // Step 3: Try to re-admit preempted sequences (priority over new)
        try_admit_preempted(output, remaining_tokens, remaining_seqs);

        // Step 4: Try to admit new sequences from waiting queue
        try_admit_waiting(output, remaining_tokens, remaining_seqs);
    } else {
        // FCFS: process in order
        schedule_prefill(output, remaining_tokens, remaining_seqs);
        schedule_decode(output, remaining_tokens, remaining_seqs);
        try_admit_preempted(output, remaining_tokens, remaining_seqs);
        try_admit_waiting(output, remaining_tokens, remaining_seqs);
    }

    return output;
}

void Scheduler::schedule_decode(SchedulerOutput& output, int32_t& remaining_tokens,
                                 int32_t& remaining_seqs) {
    for (int32_t seq_id : running_queue_) {
        if (remaining_tokens < 1 || remaining_seqs < 1) break;

        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) continue;

        Sequence& seq = it->second;
        if (seq.status != SequenceStatus::RUNNING) continue;
        if (seq.is_prefill()) continue;  // Still in prefill phase
        if (seq.next_token < 0) continue;  // No token to decode

        add_to_schedule(output, seq, 1, false);
        remaining_tokens -= 1;
        remaining_seqs -= 1;
    }
}

void Scheduler::schedule_prefill(SchedulerOutput& output, int32_t& remaining_tokens,
                                  int32_t& remaining_seqs) {
    for (int32_t seq_id : running_queue_) {
        if (remaining_tokens < 1 || remaining_seqs < 1) break;

        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) continue;

        Sequence& seq = it->second;
        if (seq.status != SequenceStatus::RUNNING) continue;
        if (!seq.is_prefill()) continue;  // Already in decode phase

        // Calculate chunk size
        int32_t remaining_prefill = seq.remaining_prefill();
        int32_t chunk_size = std::min({
            remaining_tokens,
            config_.max_prefill_tokens,
            remaining_prefill
        });

        if (chunk_size > 0) {
            add_to_schedule(output, seq, chunk_size, true);
            remaining_tokens -= chunk_size;
            remaining_seqs -= 1;
        }
    }
}

void Scheduler::try_admit_waiting(SchedulerOutput& output, int32_t& remaining_tokens,
                                   int32_t& remaining_seqs) {
    while (!waiting_queue_.empty() && remaining_tokens > 0 && remaining_seqs > 0) {
        int32_t seq_id = waiting_queue_.front();

        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) {
            waiting_queue_.pop_front();
            continue;
        }

        Sequence& seq = it->second;

        // Check if we can admit this sequence
        if (!can_admit_new_sequence()) break;

        // Check memory pressure before admitting
        int32_t blocks_needed = estimate_blocks_needed(seq, seq.prompt_len());
        if (need_preemption(blocks_needed)) break;

        // Try to allocate KV slot
        if (!admit_sequence(seq)) break;

        waiting_queue_.pop_front();
        running_queue_.push_back(seq_id);

        // Schedule first prefill chunk
        int32_t chunk_size = std::min({
            remaining_tokens,
            config_.max_prefill_tokens,
            seq.prompt_len()
        });

        if (chunk_size > 0) {
            add_to_schedule(output, seq, chunk_size, true);
            remaining_tokens -= chunk_size;
            remaining_seqs -= 1;
        }
    }
}

void Scheduler::try_admit_preempted(SchedulerOutput& output, int32_t& remaining_tokens,
                                     int32_t& remaining_seqs) {
    // Re-admit preempted sequences with priority over new requests
    // Preempted sequences need to restart from beginning (KV cache was released)
    while (!preempted_queue_.empty() && remaining_tokens > 0 && remaining_seqs > 0) {
        int32_t seq_id = preempted_queue_.front();

        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) {
            preempted_queue_.pop_front();
            continue;
        }

        Sequence& seq = it->second;

        // Check if we can admit this sequence
        if (!can_admit_new_sequence()) break;

        // Check memory pressure before admitting
        int32_t blocks_needed = estimate_blocks_needed(seq, seq.prompt_len());
        if (need_preemption(blocks_needed)) break;

        // Reset sequence state for re-prefill
        seq.prefill_pos = 0;
        seq.output_tokens.clear();
        seq.next_token = -1;

        // Try to allocate KV slot
        if (!admit_sequence(seq)) break;

        preempted_queue_.pop_front();
        running_queue_.push_back(seq_id);

        // Schedule first prefill chunk
        int32_t chunk_size = std::min({
            remaining_tokens,
            config_.max_prefill_tokens,
            seq.prompt_len()
        });

        if (chunk_size > 0) {
            add_to_schedule(output, seq, chunk_size, true);
            remaining_tokens -= chunk_size;
            remaining_seqs -= 1;
        }

        LOG(INFO) << "Re-admitted preempted sequence " << seq_id
                  << " (preemptions=" << seq.num_preemptions << ")";
    }
}

void Scheduler::add_to_schedule(SchedulerOutput& output, Sequence& seq,
                                 int32_t num_tokens, bool is_prefill) {
    ScheduledSequence sched_seq;
    sched_seq.seq_id = seq.seq_id;
    sched_seq.kv_slot = seq.kv_slot;
    sched_seq.num_tokens = num_tokens;
    sched_seq.is_prefill = is_prefill;

    if (is_prefill) {
        // Prefill: extract tokens from prompt
        sched_seq.start_pos = seq.prefill_pos;
        sched_seq.context_len = seq.prefill_pos + num_tokens;
        sched_seq.tokens.assign(
            seq.prompt_tokens.begin() + seq.prefill_pos,
            seq.prompt_tokens.begin() + seq.prefill_pos + num_tokens
        );
        output.num_prefill_tokens += num_tokens;
        output.num_prefill_seqs += 1;
    } else {
        // Decode: single token
        sched_seq.start_pos = seq.context_len();
        sched_seq.context_len = seq.context_len() + 1;
        sched_seq.tokens = {seq.next_token};
        output.num_decode_tokens += 1;
        output.num_decode_seqs += 1;
    }

    output.num_tokens += num_tokens;
    output.scheduled_seqs.push_back(std::move(sched_seq));
}

void Scheduler::update_after_forward(const SchedulerOutput& output,
                                      const std::vector<int32_t>& next_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < output.scheduled_seqs.size(); ++i) {
        const auto& sched_seq = output.scheduled_seqs[i];
        auto it = sequences_.find(sched_seq.seq_id);
        if (it == sequences_.end()) continue;

        Sequence& seq = it->second;

        if (sched_seq.is_prefill) {
            // Update prefill progress
            seq.prefill_pos += sched_seq.num_tokens;

            // If prefill just completed, record first token time
            if (!seq.is_prefill() && seq.first_token_time.time_since_epoch().count() == 0) {
                seq.first_token_time = std::chrono::steady_clock::now();
            }
        } else {
            // Decode: add the token we just processed to output
            seq.output_tokens.push_back(seq.next_token);
        }

        // Update next token from model output
        if (i < next_tokens.size()) {
            seq.next_token = next_tokens[i];
        }

        // Update KV cache context length
        if (kv_manager_ != nullptr && seq.kv_slot >= 0) {
            kv_manager_->set_context_len(seq.kv_slot, seq.context_len());
        }
    }
}

void Scheduler::finish_sequence(int32_t seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return;

    Sequence& seq = it->second;
    if (seq.status == SequenceStatus::FINISHED) return;

    seq.status = SequenceStatus::FINISHED;

    // Remove from running queue
    auto run_it = std::find(running_queue_.begin(), running_queue_.end(), seq_id);
    if (run_it != running_queue_.end()) {
        running_queue_.erase(run_it);
    }

    // Release KV slot via KVCacheManager
    if (seq.kv_slot >= 0 && kv_manager_ != nullptr) {
        kv_manager_->release_sequence_slot(seq.kv_slot);
        seq.kv_slot = -1;
    }

    finished_seqs_.push_back(seq_id);
}

const Sequence* Scheduler::get_sequence(int32_t seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return nullptr;
    return &it->second;
}

Sequence* Scheduler::get_sequence_mut(int32_t seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return nullptr;
    return &it->second;
}

bool Scheduler::all_finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return waiting_queue_.empty() && running_queue_.empty() && preempted_queue_.empty();
}

bool Scheduler::has_pending_work() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !waiting_queue_.empty() || !running_queue_.empty() || !preempted_queue_.empty();
}

int32_t Scheduler::num_waiting() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(waiting_queue_.size());
}

int32_t Scheduler::num_running() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(running_queue_.size());
}

int32_t Scheduler::num_preempted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(preempted_queue_.size());
}

int32_t Scheduler::num_finished() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(finished_seqs_.size());
}

std::vector<const Sequence*> Scheduler::get_finished_sequences() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const Sequence*> result;
    result.reserve(finished_seqs_.size());
    for (int32_t seq_id : finished_seqs_) {
        auto it = sequences_.find(seq_id);
        if (it != sequences_.end()) {
            result.push_back(&it->second);
        }
    }
    return result;
}

std::vector<const Sequence*> Scheduler::get_all_sequences() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const Sequence*> result;
    result.reserve(sequences_.size());
    for (const auto& pair : sequences_) {
        result.push_back(&pair.second);
    }
    // Sort by seq_id for consistent ordering
    std::sort(result.begin(), result.end(),
              [](const Sequence* a, const Sequence* b) {
                  return a->seq_id < b->seq_id;
              });
    return result;
}

bool Scheduler::can_admit_new_sequence() const {
    // Check if KVCacheManager has free slots
    if (kv_manager_ == nullptr) {
        return false;
    }
    return kv_manager_->has_free_slot();
}

int32_t Scheduler::estimate_blocks_needed(const Sequence& seq, int32_t additional_tokens) const {
    int32_t total_tokens = seq.context_len() + additional_tokens;
    return (total_tokens + config_.block_size - 1) / config_.block_size;
}

bool Scheduler::admit_sequence(Sequence& seq) {
    if (kv_manager_ == nullptr) {
        return false;
    }

    // Allocate KV slot via KVCacheManager
    int32_t kv_slot = kv_manager_->allocate_sequence_slot();
    if (kv_slot < 0) {
        return false;
    }

    seq.kv_slot = kv_slot;
    seq.status = SequenceStatus::RUNNING;

    // Pre-allocate KV cache blocks for prompt
    if (!kv_manager_->allocate_blocks_for_tokens(kv_slot, seq.prompt_len())) {
        // Failed to allocate blocks, release slot and return
        kv_manager_->release_sequence_slot(kv_slot);
        seq.kv_slot = -1;
        seq.status = SequenceStatus::WAITING;
        return false;
    }

    return true;
}

bool Scheduler::need_preemption(int32_t blocks_needed) const {
    if (kv_manager_ == nullptr) {
        return false;
    }
    int32_t free_blocks = kv_manager_->num_free_blocks();
    // Need preemption if free blocks < needed + min_free_blocks buffer
    return free_blocks < (blocks_needed + config_.min_free_blocks);
}

int32_t Scheduler::select_victim_sequence() const {
    // Strategy (priority order):
    // 1. Prefill sequences: select by smallest context_len (least computation invested)
    // 2. Decode sequences: select by smallest num_generated (least output produced)
    //
    // Rationale:
    // - Prefill sequences haven't produced any output yet, user impact is minimal
    // - Decode sequences are actively generating, user is waiting for output

    int32_t victim_id = -1;
    int32_t min_metric = INT32_MAX;
    bool found_prefill_victim = false;

    // First pass: look for prefill sequences (higher priority to evict)
    for (int32_t seq_id : running_queue_) {
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) continue;

        const Sequence& seq = it->second;
        if (seq.status != SequenceStatus::RUNNING) continue;

        if (seq.is_prefill()) {
            // Prefill: use context_len as metric
            int32_t ctx_len = seq.context_len();
            if (ctx_len < min_metric) {
                victim_id = seq_id;
                min_metric = ctx_len;
                found_prefill_victim = true;
            }
        }
    }

    // If found a prefill victim, return it
    if (found_prefill_victim) {
        return victim_id;
    }

    // Second pass: look for decode sequences (lower priority to evict)
    min_metric = INT32_MAX;
    for (int32_t seq_id : running_queue_) {
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) continue;

        const Sequence& seq = it->second;
        if (seq.status != SequenceStatus::RUNNING) continue;

        if (!seq.is_prefill()) {
            // Decode: use num_generated as metric
            int32_t generated = seq.num_generated();
            if (generated < min_metric) {
                victim_id = seq_id;
                min_metric = generated;
            }
        }
    }

    return victim_id;
}

int32_t Scheduler::preempt_sequences(int32_t blocks_needed) {
    int32_t num_preempted = 0;

    while (need_preemption(blocks_needed) && !running_queue_.empty()) {
        int32_t victim_id = select_victim_sequence();
        if (victim_id < 0) break;

        auto it = sequences_.find(victim_id);
        if (it == sequences_.end()) break;

        Sequence& victim = it->second;

        // Release KV resources
        if (victim.kv_slot >= 0 && kv_manager_ != nullptr) {
            kv_manager_->release_sequence_slot(victim.kv_slot);
            victim.kv_slot = -1;
        }

        // Update status
        victim.status = SequenceStatus::PREEMPTED;
        victim.num_preemptions++;

        // Remove from running queue
        auto run_it = std::find(running_queue_.begin(), running_queue_.end(), victim_id);
        if (run_it != running_queue_.end()) {
            running_queue_.erase(run_it);
        }

        // Add to preempted queue (will be rescheduled with priority)
        preempted_queue_.push_back(victim_id);

        num_preempted++;
        LOG(WARNING) << "Preempted sequence " << victim_id
                     << " (generated=" << victim.num_generated()
                     << ", preemptions=" << victim.num_preemptions << ")";
    }

    return num_preempted;
}

}  // namespace base