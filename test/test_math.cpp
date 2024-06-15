#include <glog/logging.h>
#include <gtest/gtest.h>
#include <armadillo>

TEST(test_math, add) {
  using namespace arma;
  arma::fmat f1 = "1,2,3;";
  arma::fmat f2 = "1,2,3;";
  arma::fmat f3 = f1 + f2;
  CHECK_EQ(f3.at(0), 2);
  CHECK_EQ(f3.at(1), 4);
  CHECK_EQ(f3.at(2), 6);
}
