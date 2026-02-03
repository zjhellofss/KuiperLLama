#include <iostream>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <base/base.h>
#include "base/tick.h"
#include "base/block_allocator.h"
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
    if (argc != 3) {
        std::cout << "Usage: ./llama_infer_paged <checkpoint_path> <tokenizer_path>" << std::endl;
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    // ==========================================
    // 1. 初始化模型
    // ==========================================
    std::cout << ">>> [Init] Loading model from " << checkpoint_path << std::endl;

    // 创建模型实例
    model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path,
                             checkpoint_path, false);

    auto init_status = model.init(base::DeviceType::kDeviceCUDA);
    if (!init_status) {
        std::cerr << "Model init failed: " << init_status.get_err_code() << std::endl;
        return -1;
    }

    std::cout << ">>> [Init] Model loaded successfully" << std::endl;

    // ==========================================
    // 2. 从模型配置获取参数
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

    std::cout << "\n>>> [Model Config]" << std::endl;
    std::cout << "    dim            : " << dim << std::endl;
    std::cout << "    hidden_dim     : " << hidden_dim << std::endl;
    std::cout << "    num_layers     : " << num_layers << std::endl;
    std::cout << "    num_heads      : " << num_heads << std::endl;
    std::cout << "    num_kv_heads   : " << num_kv_heads << std::endl;
    std::cout << "    head_dim       : " << head_dim << std::endl;
    std::cout << "    seq_len        : " << seq_len << std::endl;

    // ==========================================
    // 3. 准备输入句子并编码
    // ==========================================
    const std::string sentence = "Once upon a time, in a land far away,";
    std::cout << "\n>>> [Input] Encoding sentence: \"" << sentence << "\"" << std::endl;

    auto tokens = model.encode(sentence);
    if (tokens.empty()) {
        std::cerr << "Error: Failed to encode sentence" << std::endl;
        return -1;
    }

    std::cout << "    [Encode] Token count: " << tokens.size() << std::endl;

    // ==========================================
    // 4. GPU 内存规划和 Paged KV Cache 初始化
    // ==========================================
    std::cout << "\n>>> [Memory Planning] Calculating KV cache allocation..." << std::endl;

    const int block_size = 16;  // 每个物理块包含的 token 数量

    // 4.1 预留显存 (Activation + Overhead)
    const int max_seq_len = seq_len;  // 使用模型配置的 seq_len

    // Activation memory estimation (Prefill 阶段峰值)
    const int32_t kv_dim = num_kv_heads * head_dim;
    size_t activation_peak = 0;

    // 1. Hidden states
    activation_peak += (size_t)max_seq_len * dim * sizeof(float);

    // 2. Q/K/V 投影中间张量
    activation_peak += (size_t)max_seq_len * dim * sizeof(float);      // Q
    activation_peak += (size_t)max_seq_len * kv_dim * sizeof(float);   // K
    activation_peak += (size_t)max_seq_len * kv_dim * sizeof(float);   // V

    // 3. Attention scores (O(n²) - Prefill 阶段最大的内存消耗)
    activation_peak += (size_t)num_heads * max_seq_len * max_seq_len * sizeof(float);

    // 4. FFN 中间层 (SwiGLU 有 gate 和 up 两个分支)
    activation_peak += (size_t)2 * max_seq_len * hidden_dim * sizeof(float);

    // 5. 固定开销 (cuBLAS workspace, 临时 buffer 等)
    const size_t fixed_overhead = 256UL * 1024 * 1024;  // 256MB
    const size_t reserved_memory = activation_peak + fixed_overhead;

    std::cout << "    [Profile] Estimated Activation Reserve: " << reserved_memory / 1024 / 1024 << " MB" << std::endl;

    // 4.2 计算实际可用显存
    float gpu_memory_utilization = 0.90;

    size_t free_mem, total_mem;
    get_gpu_memory_info(&free_mem, &total_mem);

    size_t available_mem = (size_t)(free_mem * gpu_memory_utilization);

    if (available_mem <= reserved_memory) {
        std::cerr << "Error: Not enough memory! Available: " << available_mem / 1024 / 1024
                  << "MB, Required: " << reserved_memory / 1024 / 1024 << "MB" << std::endl;
        return -1;
    }

    size_t kv_cache_pool_size = available_mem - reserved_memory;

    // 计算单个 Block 的物理大小
    // Layout: [max_blocks, num_layers, 2 (K+V), block_size, kv_dim]
    size_t one_block_bytes = (size_t)block_size * num_layers * 2 * kv_dim * sizeof(float);

    // 计算 Block 数量
    int total_blocks = kv_cache_pool_size / one_block_bytes;

    std::cout << "\n>>> [Memory Plan]" << "\n"
              << "    Total GPU Mem   : " << total_mem / 1024 / 1024 << " MB\n"
              << "    Free GPU Mem    : " << free_mem / 1024 / 1024 << " MB\n"
              << "    Reserved Mem    : " << reserved_memory / 1024 / 1024 << " MB\n"
              << "    KV Cache Pool   : " << kv_cache_pool_size / 1024 / 1024 << " MB\n"
              << "    Block Size      : " << one_block_bytes / 1024 << " KB\n"
              << "    >>> Total Blocks: " << total_blocks << "\n"
              << "    >>> Max Context : " << total_blocks * block_size << " Tokens\n" << std::endl;

    if (total_blocks <= 0) {
        std::cerr << "Error: No memory left for KV Cache!" << std::endl;
        return -1;
    }

    // ==========================================
    // 5. 初始化 Paged KV Cache
    // ==========================================
    const int num_seqs = 1;  // 当前只支持单个序列
    int max_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;

    std::cout << ">>> [Init] Initializing Paged KV Cache..." << std::endl;
    std::cout << "    num_seqs          : " << num_seqs << std::endl;
    std::cout << "    block_size        : " << block_size << std::endl;
    std::cout << "    total_blocks      : " << total_blocks << std::endl;
    std::cout << "    max_blocks_per_seq: " << max_blocks_per_seq << std::endl;

    auto kv_init_status = model.init_paged_kv_cache(num_seqs, block_size, total_blocks, max_blocks_per_seq);
    if (!kv_init_status) {
        std::cerr << "Error: Failed to initialize paged KV cache: " << kv_init_status.get_err_code() << std::endl;
        return -1;
    }

    std::cout << ">>> [Init] Paged KV Cache initialized successfully" << std::endl;

    // ==========================================
    // 6. 开始推理：Prefill 阶段
    // ==========================================
    std::cout << "\n>>> [Inference] Starting generation..." << std::endl;

    int32_t prompt_len = static_cast<int32_t>(tokens.size());
    int32_t pos = 0;
    int32_t next = -1;
    bool is_prompt = true;
    const int total_steps = 128;  // 最大生成步数

    // 获取 prompt embedding
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

    std::vector<int32_t> generated_tokens;

    // Prefill 阶段：处理 prompt tokens
    while (pos < prompt_len - 1) {
        pos_tensor.index<int32_t>(0) = pos;

        tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
        model.predict(input, pos_tensor, is_prompt, next);

        next = tokens.at(pos + 1);
        generated_tokens.push_back(next);
        pos += 1;
    }

    std::cout << "    [Prefill] Processed " << pos << " prompt tokens" << std::endl;

    // ==========================================
    // 7. Decode 阶段：逐个生成新 token
    // ==========================================
    is_prompt = false;
    int max_new_tokens = 50;  // 最多生成 50 个新 token

    std::cout << "\n>>> [Decode] Generating tokens..." << std::endl;
    std::cout << "Generated: ";
    fflush(stdout);

    // 用于增量解码：记录上次解码的字符串长度
    std::string prev_decoded = "";

    for (int step = 0; step < max_new_tokens && pos < total_steps; ++step) {
        // 更新位置
        pos_tensor.index<int32_t>(0) = pos;

        // 使用上一步生成的 token 作为输入
        std::vector<int32_t> current_token = {next};
        const auto& token_embedding = model.embedding(current_token);
        tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);

        // 执行推理（KVCacheManager 会自动管理块分配）
        model.predict(input, pos_tensor, is_prompt, next);

        // 检查是否结束
        if (model.is_sentence_ending(next)) {
            std::cout << std::endl;
            std::cout << "    [Decode] Reached end of sentence token" << std::endl;
            break;
        }

        // 添加生成的 token
        generated_tokens.push_back(next);

        // 增量解码：解码所有已生成的 token，只输出新增部分
        std::string full_decoded = model.decode(generated_tokens);
        if (full_decoded.length() > prev_decoded.length()) {
            std::cout << full_decoded.substr(prev_decoded.length());
            fflush(stdout);
        }
        prev_decoded = full_decoded;

        pos += 1;
    }

    std::cout << std::endl;

    // ==========================================
    // 8. 输出统计信息
    // ==========================================
    std::cout << "\n>>> [Statistics]" << std::endl;
    std::cout << "    Prompt tokens    : " << prompt_len << std::endl;
    std::cout << "    Generated tokens : " << generated_tokens.size() << std::endl;
    std::cout << "    Total tokens     : " << pos + 1 << std::endl;

    std::cout << "\n>>> [Success] Paged attention inference completed!" << std::endl;

    return 0;
}