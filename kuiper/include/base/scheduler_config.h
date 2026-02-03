#ifndef KUIPER_BASE_SCHEDULER_CONFIG_H
#define KUIPER_BASE_SCHEDULER_CONFIG_H

#include <cstdint>

namespace base {

struct SchedulerConfig {
    // Maximum number of tokens to process in a single iteration
    // This limits GPU memory usage for activations
    int32_t max_num_batched_tokens = 512;

    // Maximum number of sequences in a single batch
    int32_t max_num_seqs = 256;

    // Maximum sequence length (context + generated)
    int32_t max_model_len = 4096;

    // Maximum tokens for a single prefill chunk
    // Smaller chunks allow decode requests to interleave
    int32_t max_prefill_tokens = 128; 

    // Minimum free blocks to keep as buffer
    // Prevents OOM during decode phase
    int32_t min_free_blocks = 8;

    // Block size for KV cache (must match KVCacheManager)
    int32_t block_size = 16;

    // Policy: true mean decode first , false = FCFS
    bool decode_priority = true;
};

} 

#endif  // KUIPER_BASE_SCHEDULER_CONFIG_H
