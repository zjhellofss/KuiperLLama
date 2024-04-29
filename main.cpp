#include <glog/logging.h>
#include <iostream>
#include <memory>
#include "base/alloc.h"
#include "base/buffer.h"
#include "model/llama2.h"
#include "tensor/tensor.h"

int main() {
  std::shared_ptr<base::CPUDeviceAllocator> alloc = std::make_shared<base::CPUDeviceAllocator>();
  //  Tensor tensor(DataType::kDataTypeFp32, 1, 2, 3, 4);
  //  tensor.allocate(alloc);
  //  tensor.allocate(alloc);
  //
  //  tensor.reset(DataType::kDataTypeFp32, {4, 5, 6});
  //  tensor.allocate(alloc);
  //
  //  tensor.reshape({11, 12, 13});
  //  const auto& strides = tensor.strides();

  //  char* checkpoint_path = "/home/fss/big_model/llama2_7b.bin";  // e.g. out/model.bin
  //  char* tokenizer_path = "/home/fss/big_model/tokenizer.model";
  //  model::LLama2Model model(tokenizer_path, checkpoint_path);
  //  model.init();
  //  std::string sentence = "Hi everyone";
  //  const auto& tokens = model.encode(sentence);
  //  model.forward(tokens,0);

  tensor::Tensor t1(base::DataType::kDataTypeInt32, 12);
  t1.allocate(alloc);

  int32_t* ptr = t1.ptr<int32_t>();
  for (int i = 0; i < 12; ++i) {
    *(ptr + i) = (i + 1);
  }

  std::vector<int32_t> dims{4};
  t1.reshape(dims);

  int32_t* ptr2 = t1.ptr<int32_t>();
  for (int i = 0; i < 4; ++i) {
    printf("%d\n", *(ptr2 + i));
  }
  return 0;
}