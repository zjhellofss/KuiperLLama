#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/config.h>
#include <op/matmul.h>
#include <sys/mman.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"

TEST(test_load, load_model_config) {
  std::string model_path = "./tmp/test.bin";
  int32_t fd = open(model_path.data(), O_RDONLY);
  ASSERT_NE(fd, -1);

  FILE* file = fopen(model_path.data(), "rb");
  ASSERT_NE(file, nullptr);

  auto config = model::ModelConfig{};
  fread(&config, sizeof(model::ModelConfig), 1, file);
  ASSERT_EQ(config.dim, 16);
  ASSERT_EQ(config.hidden_dim, 128);
  ASSERT_EQ(config.layer_num, 256);
}

TEST(test_load, load_model_weight) {
  std::string model_path = "./tmp/test.bin";
  int32_t fd = open(model_path.data(), O_RDONLY);
  ASSERT_NE(fd, -1);

  FILE* file = fopen(model_path.data(), "rb");
  ASSERT_NE(file, nullptr);

  auto config = model::ModelConfig{};
  fread(&config, sizeof(model::ModelConfig), 1, file);

  fseek(file, 0, SEEK_END);
  auto file_size = ftell(file);

  void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  float* weight_data =
      reinterpret_cast<float*>(static_cast<int8_t*>(data) + sizeof(model::ModelConfig));

  for (int i = 0; i < config.dim * config.hidden_dim; ++i) {
    ASSERT_EQ(*(weight_data + i), float(i));
  }
}

TEST(test_load, create_matmul) {
  std::string model_path = "./tmp/test.bin";
  int32_t fd = open(model_path.data(), O_RDONLY);
  ASSERT_NE(fd, -1);

  FILE* file = fopen(model_path.data(), "rb");
  ASSERT_NE(file, nullptr);

  auto config = model::ModelConfig{};
  fread(&config, sizeof(model::ModelConfig), 1, file);

  fseek(file, 0, SEEK_END);
  auto file_size = ftell(file);

  void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  float* weight_data =
      reinterpret_cast<float*>(static_cast<int8_t*>(data) + sizeof(model::ModelConfig));

  for (int i = 0; i < config.dim * config.hidden_dim; ++i) {
    ASSERT_EQ(*(weight_data + i), float(i));
  }
  /**                                  1
   *    1 2 3 4 5 6 ... 1024           1
   *                                   1
   *                                   1
   */
  auto wq = std::make_shared<op::MatmulLayer>(base::DeviceType::kDeviceCPU, config.dim,
                                              config.hidden_dim, false);
  float* in = new float[config.hidden_dim];
  for (int i = 0; i < config.hidden_dim; ++i) {
    in[i] = 1.f;
  }

  float* out = new float[config.dim];
  for (int i = 0; i < config.dim; ++i) {
    out[i] = 0.f;
  }
  tensor::Tensor tensor(base::DataType::kDataTypeFp32, config.hidden_dim, false, nullptr, in);
  tensor.set_device_type(base::DeviceType::kDeviceCPU);

  tensor::Tensor out_tensor(base::DataType::kDataTypeFp32, config.dim, false, nullptr, out);
  out_tensor.set_device_type(base::DeviceType::kDeviceCPU);

  wq->set_input(0, tensor);
  wq->set_output(0, out_tensor);
  wq->set_weight(0, {config.dim, config.hidden_dim}, weight_data, base::DeviceType::kDeviceCPU);
  wq->forward(); // 完成一个计算

  /** python code:
   *  w = np.arange(0,128 * 16).reshape(16, 128)
   *  input = np.ones(128)
   *  out = w@input
   */
  ASSERT_EQ(out[0], 8128);
  ASSERT_EQ(out[1], 24512);
  ASSERT_EQ(out[14], 237504);
  ASSERT_EQ(out[15], 253888);

  delete[] in;
  delete[] out;
}