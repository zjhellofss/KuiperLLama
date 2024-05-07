#include <glog/logging.h>
#include "op/encode.h"
namespace op {
std::vector<int32_t> EncodeLayer::encode(const std::string& sentence) {
  CHECK(sentence_piece_processor_ != nullptr);
  std::vector<int32_t> input_ids = sentence_piece_processor_->EncodeAsIds(sentence);
  if (has_bos_) {
    input_ids.insert(input_ids.begin(), sentence_piece_processor_->bos_id());
  }
  if (has_eos_) {
    input_ids.push_back(sentence_piece_processor_->eos_id());
  }
  return input_ids;
}

EncodeLayer::EncodeLayer() : Layer(LayerType::kLayerEncode, "Encode") {
}

EncodeLayer::EncodeLayer(
    bool has_bos, bool has_eos,
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sentence_piece_processor)
    : Layer(LayerType::kLayerEncode, "Encode"),
      has_bos_(has_bos),
      has_eos_(has_eos),
      sentence_piece_processor_(std::move(sentence_piece_processor)) {
}
}  // namespace op
