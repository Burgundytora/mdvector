﻿
#include <iostream>
#include <valarray>
#include <vector>

//
#include "Eigen/Dense"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

//
#include "mdarray.h"
#include "mdvector.h"
#include "other_method/base_expr/base_expr.h"
#include "simd/simd_function.h"
#include "time_cost.h"

//
#include "test_set.h"

using std::vector;

double val = 0.0;

template <class T>
void test_valarray() {
  std::valarray<T> data1_(1.0, total_element);
  std::valarray<T> data2_(2.0, total_element);
  std::valarray<T> data3_(0.0, total_element);
  std::valarray<T> data4_(3.0, total_element);

  TimerRecorder a("valarr");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_ * data2_;
    }

    if constexpr (do_div) {
      data3_ = data1_ / data2_;
    }

    val = data1_[0];
  }
}

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
      data4_[i][j] = 3;
    }
  }

  TimerRecorder a("** 2d");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] + data2_[i][j];
        }
      }
    }

    if constexpr (do_sub) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] - data2_[i][j];
        }
      }
    }

    if constexpr (do_mul) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] * data2_[i][j];
        }
      }
    }

    if constexpr (do_div) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] / data2_[i][j];
        }
      }
    }

    val = data1_[0][0];
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

  TimerRecorder a("vector");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] + data2_[i][j];
        }
      }
    }

    if constexpr (do_sub) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] - data2_[i][j];
        }
      }
    }

    if constexpr (do_mul) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] * data2_[i][j];
        }
      }
    }

    if constexpr (do_div) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data3_[i][j] = data1_[i][j] / data2_[i][j];
        }
      }
    }

    val = data1_[0][0];
  }
}

template <class T>
void test_mdvector_expr() {
  shape_2d test_shape = {dim1, dim2};
  vector_2d<T> data1_(test_shape);
  vector_2d<T> data2_(test_shape);
  vector_2d<T> data3_(test_shape);
  vector_2d<T> data4_(test_shape);

  // 赋值
  data1_.set_value(1);
  data2_.set_value(2);
  data4_.set_value(3);

  TimerRecorder a("mdvector");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_ * data2_;
    }

    if constexpr (do_div) {
      data3_ = data1_ / data2_;
    }
    val = data1_(0, 0);
  }
}

template <class T, size_t N1, size_t N2>
void test_mdarray_expr() {
  array_2d<T, N1, N2> data1_;
  array_2d<T, N1, N2> data2_;
  array_2d<T, N1, N2> data3_;
  array_2d<T, N1, N2> data4_;

  // 赋值
  data1_.set_value(1);
  data2_.set_value(2);
  data4_.set_value(3);

  TimerRecorder a("mdarray");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_ * data2_;
    }

    if constexpr (do_div) {
      data3_ = data1_ / data2_;
    }

    val = data1_(0, 0);
  }
}

template <class T>
void test_base_expr() {
  Array<T> data1_(total_element);
  Array<T> data2_(total_element);
  Array<T> data3_(total_element);
  Array<T> data4_(total_element);

  // 赋值
  for (size_t i = 0; i < total_element; i++) {
    data1_[i] = 1;
    data2_[i] = 2;
    data4_[i] = 3;
  }

  TimerRecorder a("expr");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_ * data2_;
    }

    if constexpr (do_div) {
      data3_ = data1_ / data2_;
    }
    val = data3_[0];
  }
}

// template <class T>
// void test_highway() {
//   SimdAllocator<T> allocator_;

//   T* data1_ = allocator_.allocate(total_element);
//   T* data2_ = allocator_.allocate(total_element);
//   T* data3_ = allocator_.allocate(total_element);
//   T* data4_ = allocator_.allocate(total_element);

//   // 赋值
//   for (size_t i = 0; i < total_element; i++) {
//     data1_[i] = 1;
//     data2_[i] = 2;
//     data4_[i] = 4;
//   }

//   TimerRecorder a("hwy 1d");

//   size_t k = 0;
//   while (k++ < loop) {
//     if constexpr (do_add) {
//       hwy_add(data1_, data2_, data3_, total_element);
//     }

//     if constexpr (do_sub) {
//       hwy_sub<T>(data1_, data2_, data3_, total_element);
//     }

//     if constexpr (do_mul) {
//       hwy_mul<T>(data1_, data2_, data3_, total_element);
//     }

//     if constexpr (do_div) {
//       hwy_div<T>(data1_, data2_, data3_, total_element);
//     }
//   }

//   allocator_.deallocate(data1_);
//   allocator_.deallocate(data2_);
//   allocator_.deallocate(data3_);
// }

template <class T>
void test_simd() {
  md::SimdAllocator<T> allocator_;

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

  TimerRecorder a("simd 1d");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      md::simd_add<T, md::AlignedPolicy>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_sub) {
      md::simd_sub<T, md::AlignedPolicy>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_mul) {
      md::simd_mul<T, md::AlignedPolicy>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_div) {
      md::simd_div<T, md::AlignedPolicy>(data1_, data2_, data3_, total_element);
    }
    val = data3_[0];
  }

  allocator_.deallocate(data1_);
  allocator_.deallocate(data2_);
  allocator_.deallocate(data3_);
}

void test_eigen_matrixd() {
  // 定义对齐的动态矩阵类型
  using AlignedMatrixXd =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, Eigen::Dynamic, Eigen::Dynamic>;

  AlignedMatrixXd data1_(dim1, dim2);
  AlignedMatrixXd data2_(dim1, dim2);
  AlignedMatrixXd data3_(dim1, dim2);
  AlignedMatrixXd data4_(dim1, dim2);

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_(i, j) = 1;
      data2_(i, j) = 2;
      data4_(i, j) = 3;
    }
  }

  TimerRecorder a("eigen");

  double temp;
  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_.cwiseProduct(data2_);
    }

    if constexpr (do_div) {
      data3_ = data1_.cwiseQuotient(data2_);
    }
    val = data3_(0, 0);
  }

  return;
}

template <class T>
void test_xarray() {
  xt::xarray<T> data1_ = xt::zeros<T>({dim1, dim2});
  xt::xarray<T> data2_ = xt::zeros<T>({dim1, dim2});
  xt::xarray<T> data3_ = xt::zeros<T>({dim1, dim2});
  xt::xarray<T> data4_ = xt::zeros<T>({dim1, dim2});

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_(i, j) = 1;
      data2_(i, j) = 2;
      data4_(i, j) = 3;
    }
  }

  TimerRecorder a("xarray");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_ * data2_;
    }

    if constexpr (do_div) {
      data3_ = data1_ / data2_;
    }
    val = data3_(0, 0);
  }
}

template <class T>
void test_xtensor() {
  xt::xtensor<T, 2> data1_ = xt::zeros<T>({dim1, dim2});
  xt::xtensor<T, 2> data2_ = xt::zeros<T>({dim1, dim2});
  xt::xtensor<T, 2> data3_ = xt::zeros<T>({dim1, dim2});
  xt::xtensor<T, 2> data4_ = xt::zeros<T>({dim1, dim2});

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_(i, j) = 1;
      data2_(i, j) = 2;
      data4_(i, j) = 3;
    }
  }

  TimerRecorder a("xtensor");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      data3_ = data1_ + data2_;
    }

    if constexpr (do_sub) {
      data3_ = data1_ - data2_;
    }

    if constexpr (do_mul) {
      data3_ = data1_ * data2_;
    }

    if constexpr (do_div) {
      data3_ = data1_ / data2_;
    }
    val = data3_(0, 0);
  }
}

int main(int args, char* argv[]) {
  md::print_simd_type();

  for (const auto& test : all_test_points) {
    loop = test.loop_;
    dim1 = test.dim1_;
    dim2 = test.dim2_;
    total_element = test.total_element_;
    total_cal = test.total_cal_;

    std::cout << "2d matrix ? matrix: " << dim1 << "*" << dim2 << "\n";

    // double
    test_simd<double>();
    test_mdvector_expr<double>();

    // test_mdarray_expr<double, dim1, dim2>();
    // 静态分派 mdarray 测试
    if (dim1 == 1 && dim2 == 4) {
      test_mdarray_expr<double, 1, 4>();
    } else if (dim1 == 1 && dim2 == 10) {
      test_mdarray_expr<double, 1, 10>();
    } else if (dim1 == 1 && dim2 == 50) {
      test_mdarray_expr<double, 1, 50>();
    } else if (dim1 == 3 && dim2 == 70) {
      test_mdarray_expr<double, 3, 70>();
    } else if (dim1 == 5 && dim2 == 100) {
      test_mdarray_expr<double, 5, 100>();
    } else if (dim1 == 10 && dim2 == 100) {
      test_mdarray_expr<double, 10, 100>();
    } else if (dim1 == 100 && dim2 == 100) {
      test_mdarray_expr<double, 100, 100>();
    }

    test_eigen_matrixd();
    test_vector<double>();
    test_norm<double>();
    test_valarray<double>();
    test_base_expr<double>();
    test_xtensor<double>();
    test_xarray<double>();
  }
  TimerRecorder::SaveSpeedResult("2d_speed_result.csv");
  std::cout << "test complete" << std::endl;

  return 0;
}