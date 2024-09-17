#ifndef KUIPER_INCLUDE_OP_ENCODE_H_
#define KUIPER_INCLUDE_OP_ENCODE_H_
#include <sentencepiece_processor.h>
#include "layer.h"
namespace op {
class EncodeLayer : public Layer {
 public:
  explicit EncodeLayer(base::DeviceType device_type);

  explicit EncodeLayer(
      base::DeviceType device_type, bool has_bos, bool has_eos,
      std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor);

  std::vector<int32_t> encode(const std::string& sentence) const;

  std::string decode(int32_t token_id) const;

  std::string decode(const std::vector<int32_t>& token_ids) const;

  int32_t eos() const;

 private:
  bool has_bos_ = true;
  bool has_eos_ = false;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_ENCODE_H_
