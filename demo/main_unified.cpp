#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <base/base.h>
#include "base/batch_metadata.h"
#include "base/kv_cache_manager.h"
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
        std::cout << "Usage: ./llama_infer_unified <checkpoint_path> <tokenizer_path> [num_seqs] [chunk_size]" << std::endl;
        std::cout << "  num_seqs: number of parallel sequences (default: 4)" << std::endl;
        std::cout << "  chunk_size: optional, 0 for auto (default: 0)" << std::endl;
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    int32_t num_seqs = (argc > 3) ? std::atoi(argv[3]) : 4;
    int32_t chunk_size = (argc > 4) ? std::atoi(argv[4]) : 0;

    // ==========================================
    // 1. Initialie model
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
    const int32_t num_heads = config->head_num_;
    const int32_t num_layers = config->layer_num_;
    const int32_t dim = config->dim_;
    const int32_t hidden_dim = config->hidden_dim_;
    const int32_t seq_len = config->seq_len_;
    const int32_t kv_dim = num_kv_heads * head_dim;

    std::cout << "\n>>> [Model Config]" << std::endl;
    std::cout << "    dim            : " << dim << std::endl;
    std::cout << "    hidden_dim     : " << hidden_dim << std::endl;
    std::cout << "    num_layers     : " << num_layers << std::endl;
    std::cout << "    num_heads      : " << num_heads << std::endl;
    std::cout << "    num_kv_heads   : " << num_kv_heads << std::endl;
    std::cout << "    head_dim       : " << head_dim << std::endl;
    std::cout << "    seq_len        : " << seq_len << std::endl;

    // ==========================================
    // 3. Prepare input sentences (multiple sequences)
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
        "What are you doing, Ouch" ,
        "Once upon a time, there was a mountain, and on the mountain, there was a temple.",
        "It's been a long day without you, my friend.",
        "Youth is not a time of life; it is a state of mind;it is not a matter of rosy cheeks, red lips and supple knees;it is a matter of the will, a quality of the imagination, a vigor of the emotions",

    };

    // Limit to requested number of sequences
    if (num_seqs < static_cast<int32_t>(sentences.size())) {
        sentences.resize(num_seqs);
    }
    int32_t actual_num_seqs = static_cast<int32_t>(sentences.size());

    std::cout << "\n>>> [Input] Encoding " << actual_num_seqs << " sequences:" << std::endl;

    std::vector<std::vector<int32_t>> all_tokens(actual_num_seqs);
    int32_t max_prompt_len = 0;
    for (int32_t i = 0; i < actual_num_seqs; ++i) {
        all_tokens[i] = model.encode(sentences[i]);
        if (all_tokens[i].empty()) {
            std::cerr << "Error: Failed to encode sentence " << i << std::endl;
            return -1;
        }
        max_prompt_len = std::max(max_prompt_len, static_cast<int32_t>(all_tokens[i].size()));
        std::cout << "    Seq " << i << ": " << all_tokens[i].size() << " tokens - \""
                  << sentences[i].substr(0, 40) << "...\"" << std::endl;
    }

    // ==========================================
    // 4. Initialize Paged KV Cache
    // ==========================================
    const int block_size = 16;

    size_t free_mem, total_mem;
    get_gpu_memory_info(&free_mem, &total_mem);

    // Calculate available blocks
    size_t one_block_bytes = (size_t)block_size * num_layers * 2 * kv_dim * sizeof(float);
    size_t reserved_memory = 512UL * 1024 * 1024;  // 512MB for activations
    size_t kv_cache_pool_size = (size_t)(free_mem * 0.9) - reserved_memory;
    int total_blocks = kv_cache_pool_size / one_block_bytes;
    int max_blocks_per_seq = (seq_len + block_size - 1) / block_size;

    std::cout << "\n>>> [Memory Plan]" << std::endl;
    std::cout << "    KV Cache Pool   : " << kv_cache_pool_size / 1024 / 1024 << " MB" << std::endl;
    std::cout << "    Total Blocks    : " << total_blocks << std::endl;
    std::cout << "    Max Context     : " << total_blocks * block_size << " Tokens" << std::endl;

    auto kv_init_status = model.init_paged_kv_cache(actual_num_seqs, block_size, total_blocks, max_blocks_per_seq);
    if (!kv_init_status) {
        std::cerr << "Error: Failed to initialize paged KV cache" << std::endl;
        return -1;
    }

    std::cout << ">>> [Init] Paged KV Cache initialized for " << actual_num_seqs << " sequences" << std::endl;

    // ==========================================
    // 5. Chunked Prefill for all sequences
    // ==========================================
    std::cout << "\n>>> [Inference] Starting chunked prefill for all sequences..." << std::endl;

    auto* kv_manager = model.get_kv_cache_manager();
    std::vector<int32_t> next_tokens(actual_num_seqs, -1);
    std::vector<int32_t> prompt_lens(actual_num_seqs);

    auto prefill_start = std::chrono::steady_clock::now();
    int32_t total_prefill_tokens = 0;

    for (int32_t seq_id = 0; seq_id < actual_num_seqs; ++seq_id) {
        const auto& tokens = all_tokens[seq_id];
        prompt_lens[seq_id] = static_cast<int32_t>(tokens.size());
        total_prefill_tokens += prompt_lens[seq_id];

        // Get embeddings
        const auto& prompt_embedding = model.embedding(tokens);
        auto [input_tokens, input_embeddings, input_token_num] = prompt_embedding;

        // Compute chunk size if auto
        int32_t seq_chunk_size = chunk_size;
        if (seq_chunk_size <= 0) {
            seq_chunk_size = model.compute_chunk_size(prompt_lens[seq_id], actual_num_seqs);
        }

        // Allocate blocks for the entire prompt
        if (!kv_manager->allocate_blocks_for_tokens(seq_id, prompt_lens[seq_id])) {
            std::cerr << "Error: Failed to allocate KV cache blocks for seq " << seq_id << std::endl;
            return -1;
        }

        // Chunked prefill
        auto prefill_status = model.forward_chunked_prefill(
            seq_id,
            input_embeddings,
            0,  // start_pos
            seq_chunk_size,
            next_tokens[seq_id]);

        if (!prefill_status) {
            std::cerr << "Error: Chunked prefill failed for seq " << seq_id << ": "
                      << prefill_status.get_err_msg() << std::endl;
            return -1;
        }
    }

    auto prefill_end = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        prefill_end - prefill_start).count();

    std::cout << "    [Prefill] Processed " << total_prefill_tokens << " tokens across "
              << actual_num_seqs << " sequences in " << prefill_ms << " ms" << std::endl;
    std::cout << "    [Prefill] Throughput: " << (total_prefill_tokens * 1000.0 / prefill_ms)
              << " tokens/sec" << std::endl;

    // ==========================================
    // 6. Batch Decode phase
    // ==========================================
    std::cout << "\n>>> [Decode] Generating tokens (batch decode)..." << std::endl;

    std::vector<std::vector<int32_t>> generated_tokens(actual_num_seqs);
    std::vector<int32_t> positions(actual_num_seqs);
    std::vector<bool> finished(actual_num_seqs, false);

    for (int32_t i = 0; i < actual_num_seqs; ++i) {
        positions[i] = prompt_lens[i];
    }

    int max_new_tokens = 50;
    int total_generated = 0;

    auto decode_start = std::chrono::steady_clock::now();

    for (int step = 0; step < max_new_tokens; ++step) {
        // Collect active sequences
        std::vector<int32_t> active_seq_ids;
        std::vector<int32_t> active_positions;
        std::vector<int32_t> active_tokens;

        for (int32_t i = 0; i < actual_num_seqs; ++i) {
            if (!finished[i]) {
                // Check for end of sentence
                if (model.is_sentence_ending(next_tokens[i])) {
                    finished[i] = true;
                    continue;
                }

                generated_tokens[i].push_back(next_tokens[i]);
                active_seq_ids.push_back(i);
                active_positions.push_back(positions[i]);
                active_tokens.push_back(next_tokens[i]);
            }
        }

        if (active_seq_ids.empty()) {
            std::cout << "    [Decode] All sequences finished" << std::endl;
            break;
        }

        // Get embeddings for all active tokens
        const auto& token_embedding = model.embedding(active_tokens);
        auto [tok_ids, tok_emb, tok_num] = token_embedding;

        // Allocate blocks for new tokens
        for (size_t i = 0; i < active_seq_ids.size(); ++i) {
            int32_t seq_id = active_seq_ids[i];
            kv_manager->allocate_blocks_for_tokens(seq_id, positions[seq_id] + 1);
        }

        // Batch decode
        std::vector<int32_t> batch_next_tokens;
        auto decode_status = model.forward_decode(active_seq_ids, active_positions, tok_emb, batch_next_tokens);
        if (!decode_status) {
            std::cerr << "Error: Batch decode failed: " << decode_status.get_err_msg() << std::endl;
            break;
        }

        // Update next tokens and positions
        for (size_t i = 0; i < active_seq_ids.size(); ++i) {
            int32_t seq_id = active_seq_ids[i];
            if (i < batch_next_tokens.size()) {
                next_tokens[seq_id] = batch_next_tokens[i];
            }
            kv_manager->set_context_len(seq_id, positions[seq_id] + 1);
            positions[seq_id] += 1;
            total_generated++;
        }
    }

    auto decode_end = std::chrono::steady_clock::now();
    double decode_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        decode_end - decode_start).count();

    // ==========================================
    // 7. Print generated text for each sequence
    // ==========================================
    std::cout << "\n>>> [Generated Text]" << std::endl;
    for (int32_t i = 0; i < actual_num_seqs; ++i) {
        std::string generated_text = model.decode(generated_tokens[i]);
        std::cout << "    Seq " << i << " (" << generated_tokens[i].size() << " tokens): "
                  << sentences[i].substr(0, 30) << "... " << generated_text << std::endl;
    }

    // ==========================================
    // 8. Statistics
    // ==========================================
    std::cout << "\n>>> [Statistics]" << std::endl;
    std::cout << "    Num sequences    : " << actual_num_seqs << std::endl;
    std::cout << "    Total prompt     : " << total_prefill_tokens << " tokens" << std::endl;
    std::cout << "    Total generated  : " << total_generated << " tokens" << std::endl;
    std::cout << "    Decode time      : " << decode_ms << " ms" << std::endl;
    if (total_generated > 0) {
        std::cout << "    Decode throughput: " << (total_generated * 1000.0 / decode_ms)
                  << " tokens/sec" << std::endl;
    }

    std::cout << "\n>>> [Success] Multi-sequence unified forward inference completed!" << std::endl;

    return 0;
}