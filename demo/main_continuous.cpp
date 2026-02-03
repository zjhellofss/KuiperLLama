#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <base/base.h>
#include "base/batch_metadata.h"
#include "base/kv_cache_manager.h"
#include "base/scheduler.h"
#include "base/scheduler_config.h"
#include "base/sequence.h"
#include "model/llama3.h"
#include "model/config.h"

void get_gpu_memory_info(size_t* free, size_t* total) {
    cudaError_t status = cudaMemGetInfo(free, total);
    if (status != cudaSuccess) {
        std::cerr << "Error: cudaMemGetInfo failed with " << cudaGetErrorString(status) << std::endl;
        return;
    }
    std::cout << ">>> [GPU] Free Memory: " << *free / (1024 * 1024) << " MB" << std::endl;
    std::cout << ">>> [GPU] Total Memory: " << *total / (1024 * 1024) << " MB" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./llama_infer_continuous <checkpoint_path> <tokenizer_path> [num_seqs] [max_batched_tokens]" << std::endl;
        std::cout << "  num_seqs: number of parallel sequences (default: 4)" << std::endl;
        std::cout << "  max_batched_tokens: max tokens per iteration (default: 512)" << std::endl;
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    int32_t num_seqs = (argc > 3) ? std::atoi(argv[3]) : 4;
    int32_t max_batched_tokens = (argc > 4) ? std::atoi(argv[4]) : 512;

    // ==========================================
    // 1. Initialize model
    // ==========================================
    std::cout << ">>> [Init] Loading model from " << checkpoint_path << std::endl;

    model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
                             checkpoint_path, false);

    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        std::cerr << "Model init failed: " << init_status.get_err_code() << std::endl;
        return -1;
    }

    std::cout << ">>> [Init] Model loaded successfully" << std::endl;

    // ==========================================
    // 2. Get model config
    // ==========================================
    const auto* config = model.config();
    if (!config) {
        std::cerr << "Error: Failed to get model config" << std::endl;
        return -1;
    }

    const int32_t head_dim = config->head_size_;
    const int32_t num_kv_heads = config->kv_head_num_;
    const int32_t num_layers = config->layer_num_;
    const int32_t dim = config->dim_;
    const int32_t seq_len = config->seq_len_;
    const int32_t kv_dim = num_kv_heads * head_dim;

    std::cout << "\n>>> [Model Config]" << std::endl;
    std::cout << "    dim            : " << dim << std::endl;
    std::cout << "    num_layers     : " << num_layers << std::endl;
    std::cout << "    num_kv_heads   : " << num_kv_heads << std::endl;
    std::cout << "    head_dim       : " << head_dim << std::endl;
    std::cout << "    seq_len        : " << seq_len << std::endl;

    // ==========================================
    // 3. Prepare input sentences
    // ==========================================
    std::vector<std::string> sentences = {
        "Once upon a time, in a land far away,",
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness.",
        "A long time ago in a galaxy far, far away,",
        "It was the best of times, it was the worst of times.",
        "To be or not to be, that is the question.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "Call me Ishmael. Some years ago, never mind how long precisely,",
        "Man, What can i say?",
        "What are you doing, Ouch",
        "Once upon a time, there was a mountain, and on the mountain, there was a temple.",
        "It's been a long day without you, my friend.",
        "Youth is not a time of life; it is a state of mind.",
    };

    if (num_seqs < static_cast<int32_t>(sentences.size())) {
        sentences.resize(num_seqs);
    }
    int32_t actual_num_seqs = static_cast<int32_t>(sentences.size());

    std::cout << "\n>>> [Input] Will process " << actual_num_seqs << " sequences" << std::endl;

    // ==========================================
    // 4. Initialize Paged KV Cache
    // ==========================================
    const int block_size = 16;

    size_t free_mem, total_mem;
    get_gpu_memory_info(&free_mem, &total_mem);

    // Calculate activation memory based on model config
    // max_seq_len for activation = max_batched_tokens (chunked prefill limits this)
    int32_t max_activation_seq_len = max_batched_tokens;
    int32_t hidden_dim = config->hidden_dim_;  // FFN intermediate dimension

    size_t activation_peak = 0;
    // 1. Hidden states
    activation_peak += (size_t)max_activation_seq_len * dim * sizeof(float);
    // 2. Q/K/V projection tensors
    activation_peak += (size_t)max_activation_seq_len * dim * sizeof(float);      // Q
    activation_peak += (size_t)max_activation_seq_len * kv_dim * sizeof(float);   // K
    activation_peak += (size_t)max_activation_seq_len * kv_dim * sizeof(float);   // V
    // 3. Attention output
    activation_peak += (size_t)max_activation_seq_len * dim * sizeof(float);
    // 4. FFN intermediate (SwiGLU has gate and up branches)
    activation_peak += (size_t)2 * max_activation_seq_len * hidden_dim * sizeof(float);
    // 5. Misc buffers (softmax, layernorm, etc.)
    activation_peak += (size_t)max_activation_seq_len * dim * sizeof(float);

    // Add safety margin (20%)
    size_t reserved_memory = (size_t)(activation_peak * 1.2);

    size_t one_block_bytes = (size_t)block_size * num_layers * 2 * kv_dim * sizeof(float);
    size_t kv_cache_pool_size = (size_t)(free_mem * 0.9) - reserved_memory;
    int total_blocks = kv_cache_pool_size / one_block_bytes;
    int max_blocks_per_seq = (seq_len + block_size - 1) / block_size;

    std::cout << "\n>>> [Memory Plan]" << std::endl;
    std::cout << "    Activation Peak    : " << activation_peak / 1024 / 1024 << " MB" << std::endl;
    std::cout << "    Reserved Memory    : " << reserved_memory / 1024 / 1024 << " MB" << std::endl;
    std::cout << "    KV Cache Pool      : " << kv_cache_pool_size / 1024 / 1024 << " MB" << std::endl;
    std::cout << "    One Block Size     : " << one_block_bytes / 1024 << " KB" << std::endl;
    std::cout << "    Total Blocks       : " << total_blocks << std::endl;
    std::cout << "    Max Batched Tokens : " << max_batched_tokens << std::endl;

    auto kv_init_status = model.init_paged_kv_cache(actual_num_seqs, block_size, total_blocks, max_blocks_per_seq);
    if (!kv_init_status) {
        std::cerr << "Error: Failed to initialize paged KV cache" << std::endl;
        return -1;
    }

    auto* kv_manager = model.get_kv_cache_manager();

    std::cout << ">>> [Init] Paged KV Cache initialized" << std::endl;

    // ==========================================
    // 5. Initialize Scheduler
    // ==========================================
    base::SchedulerConfig sched_config;
    sched_config.max_num_batched_tokens = max_batched_tokens;
    sched_config.max_num_seqs = actual_num_seqs;
    sched_config.max_model_len = seq_len;
    sched_config.max_prefill_tokens = 128;  // Chunk size for prefill
    sched_config.block_size = block_size;
    sched_config.min_free_blocks = 8;
    sched_config.decode_priority = true;

    base::Scheduler scheduler(sched_config);
    scheduler.set_kv_cache_manager(kv_manager);

    // Add all requests to scheduler
    int32_t max_new_tokens = 50;
    std::cout << "\n>>> [Input] Encoding and adding " << actual_num_seqs << " sequences:" << std::endl;
    for (int32_t i = 0; i < actual_num_seqs; ++i) {
        auto prompt_tokens = model.encode(sentences[i]);
        if (prompt_tokens.empty()) {
            std::cerr << "Error: Failed to encode sentence " << i << std::endl;
            return -1;
        }
        scheduler.add_request(prompt_tokens, max_new_tokens, sentences[i]);
        std::cout << "    Seq " << i << ": " << prompt_tokens.size() << " tokens - \""
                  << sentences[i].substr(0, 40) << "...\"" << std::endl;
    }

    // ==========================================
    // 6. Continuous Batching Main Loop
    // ==========================================
    std::cout << "\n>>> [Inference] Starting continuous batching with scheduler..." << std::endl;
    std::cout << "    Decode Priority: " << (sched_config.decode_priority ? "ON" : "OFF") << std::endl;

    int total_iterations = 0;
    int total_prefill_tokens = 0;
    int total_decode_tokens = 0;

    auto start_time = std::chrono::steady_clock::now();

    // Get allocators for BatchMetadata
    auto alloc_gpu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    while (!scheduler.all_finished()) {
        // Get scheduling decision
        auto sched_output = scheduler.schedule();
        if (!sched_output.has_work()) {
            std::cout << "    [Done] No more work to schedule" << std::endl;
            break;
        }

        // Log preempted sequences
        if (!sched_output.preempted_seq_ids.empty()) {
            std::cout << "    [Preemption] Evicted " << sched_output.preempted_seq_ids.size()
                      << " sequences due to memory pressure: ";
            for (int32_t pid : sched_output.preempted_seq_ids) {
                std::cout << pid << " ";
            }
            std::cout << std::endl;
        }

        // Build batch data from scheduler output
        std::vector<int32_t> batch_seq_ids;
        std::vector<int32_t> batch_token_counts;
        std::vector<int32_t> batch_start_positions;
        std::vector<int32_t> batch_context_lens;
        std::vector<bool> batch_is_prefill;
        std::vector<int32_t> all_tokens;

        for (const auto& sched_seq : sched_output.scheduled_seqs) {
            batch_seq_ids.push_back(sched_seq.kv_slot);  // Use kv_slot for KV cache indexing
            batch_token_counts.push_back(sched_seq.num_tokens);
            batch_start_positions.push_back(sched_seq.start_pos);
            batch_context_lens.push_back(sched_seq.context_len);
            batch_is_prefill.push_back(sched_seq.is_prefill);
            all_tokens.insert(all_tokens.end(), sched_seq.tokens.begin(), sched_seq.tokens.end());
        }

        // Allocate additional blocks for decode sequences
        for (const auto& sched_seq : sched_output.scheduled_seqs) {
            if (!sched_seq.is_prefill) {
                kv_manager->allocate_blocks_for_tokens(sched_seq.kv_slot, sched_seq.context_len);
            }
        }

        // Get embeddings
        const auto& embedding_output = model.embedding(all_tokens);
        auto [tok_ids, tok_emb, tok_num] = embedding_output;

        // Create mixed batch metadata
        auto batch_meta = base::BatchMetadata::create_mixed(
            batch_seq_ids,
            batch_token_counts,
            batch_start_positions,
            batch_context_lens,
            batch_is_prefill,
            alloc_gpu,
            alloc_cpu,
            nullptr);

        // Forward pass
        std::vector<int32_t> next_tokens;
        auto forward_status = model.forward_unified(tok_emb, batch_meta, next_tokens);
        if (!forward_status) {
            std::cerr << "Error: forward_unified failed: " << forward_status.get_err_msg() << std::endl;
            break;
        }

        // Update scheduler state
        scheduler.update_after_forward(sched_output, next_tokens);

        // Check for EOS and max tokens
        for (size_t i = 0; i < sched_output.scheduled_seqs.size(); ++i) {
            const auto& sched_seq = sched_output.scheduled_seqs[i];
            const auto* seq = scheduler.get_sequence(sched_seq.seq_id);
            if (!seq) continue;

            // Check EOS
            if (i < next_tokens.size() && model.is_sentence_ending(next_tokens[i])) {
                scheduler.finish_sequence(sched_seq.seq_id);
                continue;
            }

            // Check max tokens (only for decode phase)
            if (!sched_seq.is_prefill && seq->num_generated() >= max_new_tokens) {
                scheduler.finish_sequence(sched_seq.seq_id);
            }
        }

        // Log prefill completion
        for (const auto& sched_seq : sched_output.scheduled_seqs) {
            if (sched_seq.is_prefill) {
                const auto* seq = scheduler.get_sequence(sched_seq.seq_id);
                if (seq && !seq->is_prefill()) {
                    std::cout << "    [Prefill Done] Seq " << sched_seq.seq_id
                              << " completed prefill, starting decode" << std::endl;
                }
            }
        }

        total_prefill_tokens += sched_output.num_prefill_tokens;
        total_decode_tokens += sched_output.num_decode_tokens;
        total_iterations++;

        // Progress logging every 10 iterations
        if (total_iterations % 10 == 0) {
            std::cout << "    [Iter " << total_iterations << "] "
                      << "Batch: " << sched_output.num_tokens << " tokens "
                      << "(prefill: " << sched_output.num_prefill_tokens
                      << ", decode: " << sched_output.num_decode_tokens << ") "
                      << "| Waiting: " << scheduler.num_waiting()
                      << ", Running: " << scheduler.num_running()
                      << ", Preempted: " << scheduler.num_preempted()
                      << ", Finished: " << scheduler.num_finished()
                      << std::endl;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        end_time - start_time).count();

    // ==========================================
    // 7. Print generated text for each sequence
    // ==========================================
    std::cout << "\n>>> [Generated Text]" << std::endl;
    auto all_seqs = scheduler.get_all_sequences();
    for (const auto* seq : all_seqs) {
        std::string generated_text = model.decode(seq->output_tokens);
        std::string prompt_preview = seq->original_text.substr(0, 30);
        std::cout << "    Seq " << seq->seq_id << " (" << seq->output_tokens.size() << " tokens): "
                  << prompt_preview << "... " << generated_text << std::endl;
    }

    // ==========================================
    // 8. Statistics
    // ==========================================
    std::cout << "\n>>> [Statistics]" << std::endl;
    std::cout << "    Num sequences      : " << actual_num_seqs << std::endl;
    std::cout << "    Total iterations   : " << total_iterations << std::endl;
    std::cout << "    Total prefill tok  : " << total_prefill_tokens << std::endl;
    std::cout << "    Total decode tok   : " << total_decode_tokens << std::endl;
    std::cout << "    Total time         : " << total_ms << " ms" << std::endl;
    std::cout << "    Overall throughput : " << ((total_prefill_tokens + total_decode_tokens) * 1000.0 / total_ms)
              << " tokens/sec" << std::endl;

    std::cout << "\n>>> [Scheduler Config]" << std::endl;
    std::cout << "    max_num_batched_tokens : " << sched_config.max_num_batched_tokens << std::endl;
    std::cout << "    max_prefill_tokens     : " << sched_config.max_prefill_tokens << std::endl;
    std::cout << "    decode_priority        : " << (sched_config.decode_priority ? "true" : "false") << std::endl;

    std::cout << "\n>>> [Success] Continuous batching with scheduler completed!" << std::endl;

    return 0;
}