#include "op/encode.h"
#include <glog/logging.h>
namespace op {
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

int32_t EncodeLayer::eos() const {
  CHECK(this->sentence_piece_processor_);
  return this->sentence_piece_processor_->eos_id();
}

std::vector<int32_t> EncodeLayer::encode(const std::string& sentence) const {
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

std::string EncodeLayer::decode(int32_t token_id) const {
  CHECK(sentence_piece_processor_ != nullptr);
  std::vector<int32_t> token_ids{token_id};
  return this->sentence_piece_processor_->DecodeIds(token_ids);
}
}  // namespace op
