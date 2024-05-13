#ifndef LC_INCLUDE_OP_ENCODE_H_
#define LC_INCLUDE_OP_ENCODE_H_
#include <sentencepiece_processor.h>
#include "layer.h"
namespace op {
class EncodeLayer : public Layer {
 public:
  explicit EncodeLayer();

  explicit EncodeLayer(
      bool has_bos, bool has_eos,
      std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor);

  std::vector<int32_t> encode(const std::string& sentence) const;

  std::string decode(int32_t token_id) const;

  int32_t eos() const;

 private:
  bool has_bos_ = true;
  bool has_eos_ = false;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor_;
};
}  // namespace op
#endif  // LC_INCLUDE_OP_ENCODE_H_
