#ifndef KUIPER_INCLUDE_BASE_BASE_H_
#define KUIPER_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>
#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

namespace model {
enum class ModelBufferType {
  kInputTokens = 0,
  kInputEmbeddings = 1,
  kOutputRMSNorm = 2,
  kKeyCache = 3,
  kValueCache = 4,
  kQuery = 5,
  kInputPos = 6,
  kScoreStorage = 7,
  kOutputMHA = 8,
  kAttnOutput = 9,
  kW1Output = 10,
  kW2Output = 11,
  kW3Output = 12,
  kFFNRMSNorm = 13,
  kForwardOutput = 15,
  kForwardOutputCPU = 16,
  kSinCache = 17,
  kCosCache = 18,
  kPagedKVCache = 19,
  kPagedBlockTables = 20,
  kPagedAttentionContextLens = 21,
  // Batch buffers for multi-sequence processing
  kBatchQuery = 22,           // [num_seqs, dim]
  kBatchKey = 23,             // [num_seqs, kv_dim]
  kBatchValue = 24,           // [num_seqs, kv_dim]
  kBatchOutputMHA = 25,       // [num_seqs, dim]
  kBatchAttnOutput = 26,      // [num_seqs, dim]
  kBatchRMSNormOutput = 27,   // [num_seqs, dim]
  kBatchW1Output = 28,        // [num_seqs, hidden_dim]
  kBatchW3Output = 29,        // [num_seqs, hidden_dim]
  kBatchW2Output = 30,        // [num_seqs, dim]
  kBatchFFNRMSNorm = 31,      // [num_seqs, dim]
  kBatchInputEmbeddings = 32, // [num_seqs, dim]
  kBatchPositions = 33,       // [num_seqs] - positions for each sequence
  kBatchForwardOutput = 34,   // [num_seqs, vocab_size] - logits for each sequence
  // Unified forward buffers (for chunked prefill / continuous batching)
  kUnifiedQuery = 35,         // [num_tokens, dim]
  kUnifiedKey = 36,           // [num_tokens, kv_dim]
  kUnifiedValue = 37,         // [num_tokens, kv_dim]
  kUnifiedRMSNormOutput = 38, // [num_tokens, dim]
  kUnifiedMHAOutput = 39,     // [num_tokens, dim]
  kUnifiedAttnOutput = 40,    // [num_tokens, dim]
  kUnifiedFFNRMSNorm = 41,    // [num_tokens, dim]
  kUnifiedW1Output = 42,      // [num_tokens, hidden_dim]
  kUnifiedW3Output = 43,      // [num_tokens, hidden_dim]
  kUnifiedW2Output = 44,      // [num_tokens, dim]
};
}

namespace base {
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
};

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1,
};

inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kDataTypeFp32) {
    return sizeof(float);
  } else if (data_type == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::kDataTypeInt32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;

  NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 5,
  kKeyValueHasExist = 6,
  kInvalidArgument = 7,
};

enum class TokenizerType {
  kEncodeUnknown = -1,
  kEncodeSpe = 0,
  kEncodeBpe = 1,
};

class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t get_err_code() const;

  const std::string& get_err_msg() const;

  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = StatusCode::kSuccess;
  std::string message_;
};

namespace error {
#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)

Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_BASE_H_
