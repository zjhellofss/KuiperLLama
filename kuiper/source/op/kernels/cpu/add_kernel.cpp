#include "add_kernel.h"
#include <armadillo>
#include "base/base.h"
namespace kernel {
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);
  CHECK_EQ(input1.size(), input2.size());
  CHECK_EQ(input1.size(), output.size());

  arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false,
                        true);
  arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false,
                        true);
  arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false,
                        true);
  output_vec = input_vec1 + input_vec2;
}

}  // namespace kernel
