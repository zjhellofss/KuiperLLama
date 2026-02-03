#ifndef KUIPER_BASE_SCHEDULER_H
#define KUIPER_BASE_SCHEDULER_H

#include <deque>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <functional>
#include "base/scheduler_config.h"
#include "base/sequence.h"

namespace base {

class KVCacheManager;

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);
    ~Scheduler() = default;

    // Set the KV cache manager (required before scheduling)
    void set_kv_cache_manager(KVCacheManager* kv_manager);

    // Add a new request to the waiting queue
    // Returns the assigned sequence ID
    int32_t add_request(const std::vector<int32_t>& prompt_tokens,
                        int32_t max_new_tokens = 128,
                        const std::string& original_text = "");

    // Main scheduling function - called each iteration
    // Returns which sequences to process and how many tokens each
    SchedulerOutput schedule();

    // Update sequence state after forward pass
    // next_tokens: one token per scheduled sequence (in order)
    void update_after_forward(const SchedulerOutput& output,
                              const std::vector<int32_t>& next_tokens);

    // Mark a sequence as finished (EOS or external termination)
    void finish_sequence(int32_t seq_id);

    // Get sequence by ID (for reading output tokens, etc.)
    const Sequence* get_sequence(int32_t seq_id) const;

    // Get mutable sequence by ID
    Sequence* get_sequence_mut(int32_t seq_id);

    // Check if all sequences are finished (no waiting, no running)
    bool all_finished() const;

    // Check if there's any work to do
    bool has_pending_work() const;

    // Get statistics
    int32_t num_waiting() const;
    int32_t num_running() const;
    int32_t num_preempted() const;
    int32_t num_finished() const;

    // Get all finished sequences
    std::vector<const Sequence*> get_finished_sequences() const;

    // Get all sequences (for final output)
    std::vector<const Sequence*> get_all_sequences() const;

    // Get config
    const SchedulerConfig& config() const { return config_; }

private:
    // Check if we can admit a new sequence from waiting queue
    bool can_admit_new_sequence() const;

    // Estimate blocks needed for a sequence
    int32_t estimate_blocks_needed(const Sequence& seq, int32_t additional_tokens) const;

    // Schedule decode sequences (priority)
    void schedule_decode(SchedulerOutput& output, int32_t& remaining_tokens, int32_t& remaining_seqs);

    // Schedule prefill sequences (or chunks)
    void schedule_prefill(SchedulerOutput& output, int32_t& remaining_tokens, int32_t& remaining_seqs);

    // Try to admit new sequences from waiting queue
    void try_admit_waiting(SchedulerOutput& output, int32_t& remaining_tokens, int32_t& remaining_seqs);

    // Try to re-admit preempted sequences (priority over waiting)
    void try_admit_preempted(SchedulerOutput& output, int32_t& remaining_tokens, int32_t& remaining_seqs);

    // Move sequence from waiting to running
    bool admit_sequence(Sequence& seq);

    // Add a sequence to the scheduled output
    void add_to_schedule(SchedulerOutput& output, Sequence& seq,
                         int32_t num_tokens, bool is_prefill);

    // Preemption: evict sequences when memory is tight
    // Returns number of sequences evicted
    int32_t preempt_sequences(int32_t blocks_needed);

    // Select victim sequence for preemption (lowest priority = least generated tokens)
    int32_t select_victim_sequence() const;

    // Check if memory pressure requires preemption
    bool need_preemption(int32_t blocks_needed) const;

private:
    SchedulerConfig config_;
    KVCacheManager* kv_manager_ = nullptr;

    // Sequence storage
    std::unordered_map<int32_t, Sequence> sequences_;
    int32_t next_seq_id_ = 0;

    // Queues (store seq_ids, not Sequence objects)
    std::deque<int32_t> waiting_queue_;    // FIFO for new requests
    std::deque<int32_t> running_queue_;    // Active sequences
    std::deque<int32_t> preempted_queue_;  // Preempted sequences (priority over waiting)
    std::vector<int32_t> finished_seqs_;   // Completed sequences

    mutable std::mutex mutex_;
};

}  // namespace base

#endif  // KUIPER_BASE_SCHEDULER_H