#ifndef KUIPER_BASE_SEQUENCE_H
#define KUIPER_BASE_SEQUENCE_H

#include <cstdint>
#include <vector>
#include <string>
#include <chrono>

namespace base {

// Sequence lifecycle states
enum class SequenceStatus : uint8_t {
    WAITING = 0,    // In waiting queue, not started
    RUNNING = 1,    // Being processed (has KV cache allocated)
    PREEMPTED = 2,  // Evicted due to memory pressure, can be rescheduled
    FINISHED = 3,   // Generation complete (EOS or max length)
};

// Represents a single inference request/sequence
struct Sequence {
    // Unique identifier for this sequence
    int32_t seq_id = -1;

    // Slot index in KVCacheManager (may differ from seq_id)
    int32_t kv_slot = -1;

    // Current status
    SequenceStatus status = SequenceStatus::WAITING;

    // Input prompt tokens
    std::vector<int32_t> prompt_tokens;

    // Generated output tokens
    std::vector<int32_t> output_tokens;

    // Current position in prefill (0 = not started, prompt_len = prefill done)
    int32_t prefill_pos = 0;

    // Next token to feed (for decode phase)
    int32_t next_token = -1;

    // Request metadata
    int32_t max_new_tokens = 128;
    std::string original_text;

    // Preemption tracking
    int32_t num_preemptions = 0;  // How many times this sequence was preempted

    // Timing (optional, for metrics)
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point first_token_time;

    // Total context length = prefill_pos + output_tokens.size()
    int32_t context_len() const {
        return prefill_pos + static_cast<int32_t>(output_tokens.size());
    }

    // Check if still in prefill phase
    bool is_prefill() const {
        return prefill_pos < static_cast<int32_t>(prompt_tokens.size());
    }

    // Remaining prefill tokens
    int32_t remaining_prefill() const {
        return static_cast<int32_t>(prompt_tokens.size()) - prefill_pos;
    }

    // Total prompt length
    int32_t prompt_len() const {
        return static_cast<int32_t>(prompt_tokens.size());
    }

    // Number of generated tokens
    int32_t num_generated() const {
        return static_cast<int32_t>(output_tokens.size());
    }
};

// Information about a sequence scheduled for this iteration
struct ScheduledSequence {
    int32_t seq_id;
    int32_t kv_slot;
    int32_t num_tokens;      // Tokens to process this iteration
    int32_t start_pos;       // Starting position
    int32_t context_len;     // Context length after processing
    bool is_prefill;         // true = prefill, false = decode
    std::vector<int32_t> tokens;  // Actual token IDs to process
};

// Output of scheduler.schedule() call
struct SchedulerOutput {
    // Sequences scheduled for this iteration
    std::vector<ScheduledSequence> scheduled_seqs;

    // Sequences preempted this iteration (for logging/metrics)
    std::vector<int32_t> preempted_seq_ids;

    // Total tokens in this batch
    int32_t num_tokens = 0;

    // Breakdown
    int32_t num_prefill_tokens = 0;
    int32_t num_decode_tokens = 0;
    int32_t num_prefill_seqs = 0;
    int32_t num_decode_seqs = 0;

    // True if there's work to do
    bool has_work() const { return !scheduled_seqs.empty(); }

    // True if batch contains mixed prefill + decode
    bool is_mixed() const { return num_prefill_seqs > 0 && num_decode_seqs > 0; }
};

}  // namespace base

#endif  // KUIPER_BASE_SEQUENCE_H
