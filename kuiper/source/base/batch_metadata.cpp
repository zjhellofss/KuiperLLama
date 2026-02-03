/**                                                                                                                                                                                                                                            
                                                                                                                                                                 
   * @author    jintang-coder                                                                                                                                                                                                                                                                                                                                                                                                                                                 
   *   - BatchMetadata: Token-to-sequence mapping for GPU                                                                                                         
   *                                                                                                                                                                                                                                    
   */
//  BatchMetadata 中 tokens 的拆解与映射                                                                                                                            
                                                                                                                                                                  
//   核心数据结构                                                                                                                                                    
                                                                                                                                                                  
//   struct BatchMetadata {                                                                                                                                          
//     int32_t num_tokens;           // 总token数                                                                                                                    
//     int32_t num_seqs;             // 序列数                                                                                                                       
                                                                                                                                                                  
//     // 关键映射数组 (长度 = num_tokens)                                                                                                                           
//     tensor::Tensor seq_ids;       // [num_tokens] 每个token属于哪个序列                                                                                           
//     tensor::Tensor positions;     // [num_tokens] 每个token在序列中的位置                                                                                         
                                                                                                                                                                  
//     // 序列边界 (长度 = num_seqs + 1)                                                                                                                             
//     tensor::Tensor seq_start_locs; // [num_seqs+1] 每个序列在token数组中的起始位置                                                                                
//   };                                                                                                                                                              
                                                                                                                                                                  
//   ---                                                                                                                                                             
//   具体例子：Mixed Batch (Prefill + Decode 混合)                                                                                                                   
                                                                                                                                                                  
//   假设调度器输出：                                                                                                                                                
//   Seq0 (kv_slot=0): prefill, 10 tokens, start_pos=0                                                                                                               
//   Seq1 (kv_slot=1): decode,   1 token,  start_pos=25                                                                                                              
//   Seq2 (kv_slot=2): prefill,  5 tokens, start_pos=0                                                                                                               
                                                                                                                                                                  
//   输入参数 (来自 Scheduler)                                                                                                                                       
                                                                                                                                                                  
//   seq_ids_in      = [0, 1, 2]        // kv_slot                                                                                                                   
//   token_counts    = [10, 1, 5]       // 每个序列的token数                                                                                                         
//   start_positions = [0, 25, 0]       // 每个序列的起始位置                                                                                                        
//   context_lens    = [10, 26, 5]      // 处理后的context长度                                                                                                       
//   is_prefill      = [true, false, true]                                                                                                                           
                                                                                                                                                                  
//   create_mixed() 构建过程                                                                                                                                         
                                                                                                                                                                  
//   // 1. 计算总token数                                                                                                                                             
//   num_tokens = 10 + 1 + 5 = 16                                                                                                                                    
//   num_seqs = 3                                                                                                                                                    
                                                                                                                                                                  
//   // 2. 构建 seq_start_locs (序列边界)                                                                                                                            
//   seq_start_locs = [0, 10, 11, 16]                                                                                                                                
//   //                │   │   │   └── 结束位置                                                                                                                      
//   //                │   │   └── seq2 起始 (offset=11)                                                                                                             
//   //                │   └── seq1 起始 (offset=10)                                                                                                                 
//   //                └── seq0 起始 (offset=0)                                                                                                                      
                                                                                                                                                                  
//   // 3. 展开 seq_ids (每个token属于哪个序列)                                                                                                                      
//   seq_ids = [0,0,0,0,0,0,0,0,0,0,  1,  2,2,2,2,2]                                                                                                                 
//   //        └──── seq0: 10个 ────┘  │  └─ seq2: 5个 ─┘                                                                                                            
//   //                               seq1: 1个                                                                                                                      
                                                                                                                                                                  
//   // 4. 展开 positions (每个token的位置)                                                                                                                          
//   positions = [0,1,2,3,4,5,6,7,8,9,  25,  0,1,2,3,4]                                                                                                              
//   //          └── seq0: pos 0-9 ──┘   │   └─ seq2: pos 0-4 ─┘                                                                                                     
//   //                                 seq1: pos 25                                                                                                                 
                                                      






#include "base/batch_metadata.h"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <algorithm>
#include <numeric>

namespace base {

bool BatchMetadata::has_prefill() const {
  for (bool is_pf : is_prefill_cpu) {
    if (is_pf) return true;
  }
  return false;
}

bool BatchMetadata::has_decode() const {
  for (bool is_pf : is_prefill_cpu) {
    if (!is_pf) return true;
  }
  return false;
}

int32_t BatchMetadata::max_prefill_len() const {
  int32_t max_len = 0;
  for (int32_t i = 0; i < num_seqs; ++i) {
    if (is_prefill_cpu[i]) {
      int32_t len = seq_start_locs_cpu[i + 1] - seq_start_locs_cpu[i];
      max_len = std::max(max_len, len);
    }
  }
  return max_len;
}

int32_t BatchMetadata::max_context_len() const {
  if (seq_lens_cpu.empty()) return 0;
  return *std::max_element(seq_lens_cpu.begin(), seq_lens_cpu.end());
}

int32_t BatchMetadata::get_num_prefill_tokens() const {
  int32_t count = 0;
  for (int32_t i = 0; i < num_seqs; ++i) {
    if (is_prefill_cpu[i]) {
      count += seq_start_locs_cpu[i + 1] - seq_start_locs_cpu[i];
    }
  }
  return count;
}

int32_t BatchMetadata::get_num_decode_tokens() const {
  int32_t count = 0;
  for (int32_t i = 0; i < num_seqs; ++i) {
    if (!is_prefill_cpu[i]) {
      count += seq_start_locs_cpu[i + 1] - seq_start_locs_cpu[i];
    }
  }
  return count;
}

std::vector<int32_t> BatchMetadata::get_last_token_indices() const {
  std::vector<int32_t> indices(num_seqs);
  for (int32_t i = 0; i < num_seqs; ++i) {
    indices[i] = seq_start_locs_cpu[i + 1] - 1;
  }
  return indices;
}

void BatchMetadata::sync_to_device(std::shared_ptr<DeviceAllocator> alloc_gpu,
                                   std::shared_ptr<DeviceAllocator> alloc_cpu,
                                   void* stream) {
  // Allocate GPU tensors
  seq_ids = tensor::Tensor(DataType::kDataTypeInt32, num_tokens, true, alloc_gpu);
  positions = tensor::Tensor(DataType::kDataTypeInt32, num_tokens, true, alloc_gpu);
  seq_start_locs = tensor::Tensor(DataType::kDataTypeInt32, num_seqs + 1, true, alloc_gpu);

  seq_ids.set_device_type(DeviceType::kDeviceCUDA);
  positions.set_device_type(DeviceType::kDeviceCUDA);
  seq_start_locs.set_device_type(DeviceType::kDeviceCUDA);

  // Copy data to GPU
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

  cudaMemcpyAsync(seq_ids.ptr<int32_t>(), seq_ids_cpu.data(),
                  num_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream);

  cudaMemcpyAsync(positions.ptr<int32_t>(), positions_cpu.data(),
                  num_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream);

  cudaMemcpyAsync(seq_start_locs.ptr<int32_t>(), seq_start_locs_cpu.data(),
                  (num_seqs + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, cuda_stream);
}

BatchMetadata BatchMetadata::create_decode(
    const std::vector<int32_t>& seq_ids_in,
    const std::vector<int32_t>& positions_in,
    const std::vector<int32_t>& context_lens,
    std::shared_ptr<DeviceAllocator> alloc_gpu,
    std::shared_ptr<DeviceAllocator> alloc_cpu,
    void* stream) {
  CHECK_EQ(seq_ids_in.size(), positions_in.size());
  CHECK_EQ(seq_ids_in.size(), context_lens.size());

  BatchMetadata meta;
  meta.num_tokens = static_cast<int32_t>(seq_ids_in.size());
  meta.num_seqs = static_cast<int32_t>(seq_ids_in.size());

  // For decode, each sequence has exactly 1 token
  meta.seq_ids_cpu = seq_ids_in;
  meta.positions_cpu = positions_in;
  meta.seq_lens_cpu = context_lens;

  // Build seq_start_locs: [0, 1, 2, ..., num_seqs]
  meta.seq_start_locs_cpu.resize(meta.num_seqs + 1);
  for (int32_t i = 0; i <= meta.num_seqs; ++i) {
    meta.seq_start_locs_cpu[i] = i;
  }

  // All sequences are in decode phase
  meta.is_prefill_cpu.resize(meta.num_seqs, false);

  meta.sync_to_device(alloc_gpu, alloc_cpu, stream);
  return meta;
}

BatchMetadata BatchMetadata::create_prefill(
    int32_t seq_id,
    int32_t start_pos,
    int32_t num_tokens,
    int32_t context_len,
    std::shared_ptr<DeviceAllocator> alloc_gpu,
    std::shared_ptr<DeviceAllocator> alloc_cpu,
    void* stream) {
  CHECK_GT(num_tokens, 0);

  BatchMetadata meta;
  meta.num_tokens = num_tokens;
  meta.num_seqs = 1;

  // All tokens belong to the same sequence
  meta.seq_ids_cpu.resize(num_tokens, seq_id);

  // Positions: [start_pos, start_pos+1, ..., start_pos+num_tokens-1]
  meta.positions_cpu.resize(num_tokens);
  for (int32_t i = 0; i < num_tokens; ++i) {
    meta.positions_cpu[i] = start_pos + i;
  }

  meta.seq_start_locs_cpu = {0, num_tokens};
  meta.seq_lens_cpu = {context_len};
  meta.is_prefill_cpu = {true};

  meta.sync_to_device(alloc_gpu, alloc_cpu, stream);
  return meta;
}

BatchMetadata BatchMetadata::create_mixed(
    const std::vector<int32_t>& seq_ids_in,
    const std::vector<int32_t>& token_counts,
    const std::vector<int32_t>& start_positions,
    const std::vector<int32_t>& context_lens,
    const std::vector<bool>& is_prefill,
    std::shared_ptr<DeviceAllocator> alloc_gpu,
    std::shared_ptr<DeviceAllocator> alloc_cpu,
    void* stream) {
  int32_t num_seqs = static_cast<int32_t>(seq_ids_in.size());
  CHECK_EQ(token_counts.size(), num_seqs);
  CHECK_EQ(start_positions.size(), num_seqs);
  CHECK_EQ(context_lens.size(), num_seqs);
  CHECK_EQ(is_prefill.size(), num_seqs);

  BatchMetadata meta;
  meta.num_seqs = num_seqs;
  meta.num_tokens = std::accumulate(token_counts.begin(), token_counts.end(), 0);

  meta.seq_ids_cpu.reserve(meta.num_tokens);
  meta.positions_cpu.reserve(meta.num_tokens);
  meta.seq_start_locs_cpu.resize(num_seqs + 1);
  meta.seq_lens_cpu = context_lens;
  meta.is_prefill_cpu = is_prefill;

  int32_t offset = 0;
  for (int32_t i = 0; i < num_seqs; ++i) {
    meta.seq_start_locs_cpu[i] = offset;

    for (int32_t t = 0; t < token_counts[i]; ++t) {
      meta.seq_ids_cpu.push_back(seq_ids_in[i]);
      meta.positions_cpu.push_back(start_positions[i] + t);
    }

    offset += token_counts[i];
  }
  meta.seq_start_locs_cpu[num_seqs] = offset;

  meta.sync_to_device(alloc_gpu, alloc_cpu, stream);
  return meta;
}

}  // namespace base