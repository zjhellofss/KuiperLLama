#ifndef LC_INCLUDE_OP_ENCODE_LAYER_H_
#define LC_INCLUDE_OP_ENCODE_LAYER_H_
#include <sentencepiece_processor.h>
#include "layer.h"
namespace op {
class EncodeLayer : public LayerFp32 {
 public:
  explicit EncodeLayer();

  explicit EncodeLayer(
      bool has_bos, bool has_eos,
      std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor);

  std::vector<int32_t> encode(const std::string& sentence);

 private:
  bool has_bos_ = true;
  bool has_eos_ = false;
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor_;
};
}  // namespace op
#endif  // LC_INCLUDE_OP_ENCODE_LAYER_H_
