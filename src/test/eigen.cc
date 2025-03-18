#include "Eigen/Dense"
//

#include "test_set.h"

void test_eigen_matrixd() {
  Eigen::MatrixXd data1_ = Eigen::MatrixXd::Zero(dim1, dim2);
  Eigen::MatrixXd data2_ = Eigen::MatrixXd::Zero(dim1, dim2);
  Eigen::MatrixXd data3_ = Eigen::MatrixXd::Zero(dim1, dim2);
  Eigen::MatrixXd data4_ = Eigen::MatrixXd::Zero(dim1, dim2);

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_(i, j) = 1;
      data2_(i, j) = 2;
      data4_(i, j) = 3;
    }
  }

  TimerRecorder a("double: eigen matrix");

  double temp;
  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_ - data4_;
    }

    if constexpr (do_mul) {
      data3_ = data1_.cwiseProduct(data2_).cwiseProduct(data4_);
    }

    if constexpr (do_div) {
      data3_ = data1_.cwiseQuotient(data2_).cwiseQuotient(data4_);
    }
  }

  return;
}

int main(int args, char* argv[]) {
  std::cout << "2d matrix ? matrix: " << dim1 << "*" << dim2 << "\n";

  // double
  test_eigen_matrixd();

  std::cout << "test complete" << std::endl;

  return 0;
}
