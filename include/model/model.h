#ifndef LC_INCLUDE_MODEL_MODEL_H_
#define LC_INCLUDE_MODEL_MODEL_H_
#include <string>
#include "tensor/tensor.h"

class Model {
 public:
  explicit Model(std::string token_path, std::string model_path);

  virtual void Init() = 0;

 private:
  std::string token_path;

};
#endif  // LC_INCLUDE_MODEL_MODEL_H_
