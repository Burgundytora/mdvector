
#include <iostream>
#include <format>
#include <vector>
using std::vector;

#include "src/header/avx2.h"
#include "src/header/normal.h"
#include "src/header/time_cost.h"
#include "src/header/mdvector.hpp"

#include "Eigen/Dense"

constexpr bool do_add = true;
constexpr bool do_sub = false;
constexpr bool do_mul = false;
constexpr bool do_div = false;

size_t loop = 1000000;
size_t dim1 = 10;
size_t dim2 = 50;
size_t total_element = dim1 * dim2;

template <class T>
void test_norm() {
  T** data1_ = new T*[dim1];
  T** data2_ = new T*[dim1];
  T** data3_ = new T*[dim1];
  T** data4_ = new T*[dim1];

  for (size_t i = 0; i < dim1; i++) {
    data1_[i] = new T[dim2];
    data2_[i] = new T[dim2];
    data3_[i] = new T[dim2];
    data4_[i] = new T[dim2];
  }

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_[i][j] = 1;
      data2_[i][j] = 2;
    }
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": norm");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] + data2_[i][j] + data4_[i][j];
        }
      }
    }

    if constexpr (do_sub) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] - data2_[i][j] - data4_[i][j];
        }
      }
    }

    if constexpr (do_mul) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] * data2_[i][j] * data4_[i][j];
        }
      }
    }

    if constexpr (do_div) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] / data2_[i][j] / data4_[i][j];
        }
      }
    }
  }

  delete[] data1_;
  delete[] data2_;
  delete[] data3_;
}

template <class T>
void test_vector() {
  vector<vector<T>> data1_;
  vector<vector<T>> data2_;
  vector<vector<T>> data3_;
  vector<vector<T>> data4_;

  for (size_t i = 0; i < dim1; i++) {
    data1_.push_back(vector<T>(dim2, 1));
    data2_.push_back(vector<T>(dim2, 2));
    data3_.push_back(vector<T>(dim2, 0));
    data4_.push_back(vector<T>(dim2, 3));
  }

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_[i][j] = 1;
      data2_[i][j] = 2;
      data4_[i][j] = 3;
    }
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": vector");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] + data2_[i][j] + data4_[i][j];
        }
      }
    }

    if constexpr (do_sub) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] - data2_[i][j] - data4_[i][j];
        }
      }
    }

    if constexpr (do_mul) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] * data2_[i][j] * data4_[i][j];
        }
      }
    }

    if constexpr (do_div) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] / data2_[i][j] / data4_[i][j];
        }
      }
    }
  }
}

template <class T>
void test_mdvector_expr() {
  MDShape_2d test_shape = {dim1, dim2};
  MDVector<T, 2> data1_(test_shape);
  MDVector<T, 2> data2_(test_shape);
  MDVector<T, 2> data3_(test_shape);
  MDVector<T, 2> data4_(test_shape);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_.data_[i] = 1;
    data2_.data_[i] = 2;
    data4_.data_[i] = 3;
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": mdvector expr");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_ + data4_;
    }

    // if constexpr (do_sub) {
    //   data3_ = data1_ - data2_ - data4_;
    // }

    // if constexpr (do_mul) {
    //   data3_ = data1_ * data2_ * data4_;
    // }

    // if constexpr (do_div) {
    //   data3_ = data1_ / data2_ / data4_;
    // }
  }
}

template <class T>
void test_mdvector_fun() {
  MDShape_2d test_shape = {dim1, dim2};
  MDVector<T, 2> data1_(test_shape);
  MDVector<T, 2> data2_(test_shape);
  MDVector<T, 2> data3_(test_shape);
  MDVector<T, 2> data4_(test_shape);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_.data_[i] = 1;
    data2_.data_[i] = 2;
    data4_.data_[i] = 3;
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": mdvector fun");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_.equal_a_add_b(data1_, data2_);
      data3_.equal_a_add_b(data3_, data4_);
    }

    if constexpr (do_sub) {
      data3_.equal_a_sub_b(data1_, data2_);
      data3_.equal_a_sub_b(data3_, data4_);
    }

    if constexpr (do_mul) {
      data3_.equal_a_mul_b(data1_, data2_);
      data3_.equal_a_mul_b(data3_, data4_);
    }

    if constexpr (do_div) {
      data3_.equal_a_div_b(data1_, data2_);
      data3_.equal_a_div_b(data3_, data4_);
    }
  }
}

template <class T>
void test_avx2() {
  AlignedAllocator<T> allocator_;

  T* data1_ = allocator_.allocate(total_element);
  T* data2_ = allocator_.allocate(total_element);
  T* data3_ = allocator_.allocate(total_element);
  T* data4_ = allocator_.allocate(total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_[i] = 1;
    data2_[i] = 2;
    data4_[i] = 4;
  }

  TimerRecorder a(std::string(typeid(T).name()) + ": avx2");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      avx2_add<T>(data1_, data2_, data3_, total_element);
      avx2_add<T>(data4_, data3_, data3_, total_element);
    }

    if constexpr (do_sub) {
      avx2_sub<T>(data1_, data2_, data3_, total_element);
      avx2_sub<T>(data4_, data3_, data3_, total_element);
    }

    if constexpr (do_mul) {
      avx2_mul<T>(data1_, data2_, data3_, total_element);
      avx2_mul<T>(data4_, data3_, data3_, total_element);
    }

    if constexpr (do_div) {
      avx2_div<T>(data1_, data2_, data3_, total_element);
      avx2_div<T>(data4_, data3_, data3_, total_element);
    }
  }

  allocator_.deallocate(data1_);
  allocator_.deallocate(data2_);
  allocator_.deallocate(data3_);
}

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
      data3_ = data1_ + data2_ + data4_;
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
  test_vector<double>();
  test_norm<double>();
  test_eigen_matrixd();
  test_avx2<double>();
  test_mdvector_expr<double>();
  test_mdvector_fun<double>();

  std::cout << "test complete" << std::endl;

  return 0;
}