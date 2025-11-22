
#include <valarray>
#include <execution>

//
#include "test_set.h"
#include "time_cost.h"

//
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

//
#define XTENSOR_USE_XSIMD
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"

//
#include "include_md.h"
#include "other_method/base_expr/base_expr.h"

using std::vector;

double val = 0.0;

template <class T>
void test_valarray() {
  std::valarray<T> data1_(1.0, total_element);
  std::valarray<T> data2_(2.0, total_element);
  std::valarray<T> data3_(3.0, total_element);
  std::valarray<T> data4_(4.0, total_element);
  std::valarray<T> data_res(0.0, total_element);

  {
    TimerRecorder a("valarr");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data1_ + data1_ - data2_ * data3_ / data4_;
    }
  }
  val = data_res[0];
}

template <class T>
void test_norm() {
  T** data1_ = new T*[dim1];
  T** data2_ = new T*[dim1];
  T** data3_ = new T*[dim1];
  T** data4_ = new T*[dim1];
  T** data_res = new T*[dim1];

  for (int i = 0; i < dim1; i++) {
    data1_[i] = new T[dim2];
    data2_[i] = new T[dim2];
    data3_[i] = new T[dim2];
    data4_[i] = new T[dim2];
    data_res[i] = new T[dim2];
  }

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_[i][j] = 1.0;
      data2_[i][j] = 2.0;
      data3_[i][j] = 3.0;
      data4_[i][j] = 4.0;
      data_res[i][j] = 0.0;
    }
  }

  {
    TimerRecorder a("** 2d");

    size_t k = 0;
    while (k++ < loop) {
      for (int i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data_res[i][j] = data1_[i][j] + data1_[i][j] - data2_[i][j] * data3_[i][j] / data4_[i][j];
        }
      }
    }
  }

  val = data_res[0][0];

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
  vector<vector<T>> data_res;

  for (int i = 0; i < dim1; i++) {
    data1_.push_back(vector<T>(dim2, 1));
    data2_.push_back(vector<T>(dim2, 2));
    data3_.push_back(vector<T>(dim2, 0));
    data4_.push_back(vector<T>(dim2, 3));
    data_res.push_back(vector<T>(dim2, 3));
  }

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_[i][j] = 1.0;
      data2_[i][j] = 2.0;
      data3_[i][j] = 3.0;
      data4_[i][j] = 4.0;
      data_res[i][j] = 0.0;
    }
  }

  {
    TimerRecorder a("vector");

    size_t k = 0;
    while (k++ < loop) {
      for (int i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          data_res[i][j] = data1_[i][j] + data1_[i][j] - data2_[i][j] * data3_[i][j] / data4_[i][j];
        }
      }
    }
  }

  val = data_res[0][0];
}

template <class T>
void test_transform() {
  vector<T> data1_(dim1 * dim2);
  vector<T> data2_(dim1 * dim2);
  vector<T> data3_(dim1 * dim2);
  vector<T> data4_(dim1 * dim2);
  vector<T> data_res(dim1 * dim2);

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_[i * dim2 + j] = 1.0;
      data2_[i * dim2 + j] = 2.0;
      data3_[i * dim2 + j] = 3.0;
      data4_[i * dim2 + j] = 4.0;
      data_res[i * dim2 + j] = 0.0;
    }
  }

  {
    TimerRecorder a("transform");

    size_t k = 0;
    while (k++ < loop) {
      std::transform(std::execution::unseq, data1_.begin(), data1_.end(), data2_.begin(), data3_.begin(),
                     std::plus<>());
      std::transform(std::execution::unseq, data1_.begin(), data1_.end(), data2_.begin(), data3_.begin(),
                     std::minus<>());
      std::transform(std::execution::unseq, data1_.begin(), data1_.end(), data2_.begin(), data3_.begin(),
                     std::multiplies<>());
      std::transform(std::execution::unseq, data1_.begin(), data1_.end(), data2_.begin(), data3_.begin(),
                     std::divides<>());
    }
  }

  val = data1_[0];
}

template <class T>
void test_mdvector_expr() {
  md::shape<2> test_shape = {dim1, dim2};
  md::vector<T, 2> data1_(test_shape);
  md::vector<T, 2> data2_(test_shape);
  md::vector<T, 2> data3_(test_shape);
  md::vector<T, 2> data4_(test_shape);
  md::vector<T, 2> data_res(test_shape);

  // 赋值
  data1_.fill(1.0);
  data2_.fill(2.0);
  data3_.fill(3.0);
  data4_.fill(4.0);
  data_res.fill(0.0);

  {
    TimerRecorder a("mdvector");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data1_ + data1_ - data2_ * data3_ / data4_;
    }
    val = data_res(0, 0);
  }
}

template <class T, size_t N1, size_t N2>
void test_mdarray_expr() {
  md::array<T, std::layout_right, N1, N2> data1_;
  md::array<T, std::layout_right, N1, N2> data2_;
  md::array<T, std::layout_right, N1, N2> data3_;
  md::array<T, std::layout_right, N1, N2> data4_;
  md::array<T, std::layout_right, N1, N2> data_res;

  // 赋值
  data1_.fill(1.0);
  data2_.fill(2.0);
  data3_.fill(3.0);
  data4_.fill(4.0);
  data_res.fill(0.0);

  {
    TimerRecorder a("mdarray");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data1_ + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0);
}

template <class T, size_t N1, size_t N2>
void test_mdinpvec_expr() {
  md::inplace_vector<T, 2, N1 * N2 * 4 / 3, std::layout_right> data1_;
  md::inplace_vector<T, 2, N1 * N2 * 4 / 3, std::layout_right> data2_;
  md::inplace_vector<T, 2, N1 * N2 * 4 / 3, std::layout_right> data3_;
  md::inplace_vector<T, 2, N1 * N2 * 4 / 3, std::layout_right> data4_;
  md::inplace_vector<T, 2, N1 * N2 * 4 / 3, std::layout_right> data_res;

  data1_.set_shape(N1, N2);
  data2_.set_shape(N1, N2);
  data3_.set_shape(N1, N2);
  data4_.set_shape(N1, N2);
  data_res.set_shape(N1, N2);

  // 赋值
  data1_.fill(1.0);
  data2_.fill(2.0);
  data3_.fill(3.0);
  data4_.fill(4.0);
  data_res.fill(0.0);

  {
    TimerRecorder a("mdinpvec");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data1_ + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0);
}

template <class T>
void test_base_expr() {
  Array<T> data1_(total_element);
  Array<T> data2_(total_element);
  Array<T> data3_(total_element);
  Array<T> data4_(total_element);
  Array<T> data_res(total_element);

  // 赋值
  for (int i = 0; i < total_element; i++) {
    data1_[i] = 1.0;
    data2_[i] = 2.0;
    data3_[i] = 3.0;
    data4_[i] = 4.0;
    data_res[i] = 0.0;
  }

  {
    TimerRecorder a("expr");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data1_ + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res[0];
}

// template <class T>
// void test_highway() {
//   simd_allocator<T> allocator_;

//   T* data1_ = allocator_.allocate(total_element);
//   T* data2_ = allocator_.allocate(total_element);
//   T* data3_ = allocator_.allocate(total_element);
//   T* data4_ = allocator_.allocate(total_element);

//   // 赋值
//   for (int i = 0; i < total_element; i++) {
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
  md::simd_allocator<T> allocator_;

  T* data1_ = allocator_.allocate(total_element);
  T* data2_ = allocator_.allocate(total_element);
  T* data3_ = allocator_.allocate(total_element);
  T* data4_ = allocator_.allocate(total_element);
  T* data_res = allocator_.allocate(total_element);

  // 赋值
  for (int i = 0; i < total_element; i++) {
    data1_[i] = 1.0;
    data2_[i] = 2.0;
    data3_[i] = 3.0;
    data4_[i] = 4.0;
    data_res[i] = 0.0;
  }

  {
    TimerRecorder a("simd 1d");

    size_t k = 0;
    while (k++ < loop) {
      md::simd_add<T, md::aligned_policy>(data1_, data2_, data3_, total_element);
      md::simd_sub<T, md::aligned_policy>(data1_, data2_, data3_, total_element);
      md::simd_mul<T, md::aligned_policy>(data1_, data2_, data3_, total_element);
      md::simd_div<T, md::aligned_policy>(data1_, data2_, data3_, total_element);
    }
  }

  val = data_res[0];

  allocator_.deallocate(data1_);
  allocator_.deallocate(data2_);
  allocator_.deallocate(data3_);
}

void test_eigen_tensor() {
  // 定义对齐的动态矩阵类型
  Eigen::Tensor<double, 2> data1_(
      Eigen::array<Eigen::Index, 2>{static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2)});
  Eigen::Tensor<double, 2> data2_(
      Eigen::array<Eigen::Index, 2>{static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2)});
  Eigen::Tensor<double, 2> data3_(
      Eigen::array<Eigen::Index, 2>{static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2)});
  Eigen::Tensor<double, 2> data4_(
      Eigen::array<Eigen::Index, 2>{static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2)});
  Eigen::Tensor<double, 2> data_res(
      Eigen::array<Eigen::Index, 2>{static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2)});

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_(i, j) = 1.0;
      data2_(i, j) = 2.0;
      data3_(i, j) = 3.0;
      data4_(i, j) = 4.0;
      data_res(i, j) = 0.0;
    }
  }

  {
    TimerRecorder a("eigen");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data1_ + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0);
}

template <class T>
void test_xarray() {
  xt::xarray<T> data1_ = xt::zeros<T>({dim1, dim2});
  xt::xarray<T> data2_ = xt::zeros<T>({dim1, dim2});
  xt::xarray<T> data3_ = xt::zeros<T>({dim1, dim2});
  xt::xarray<T> data4_ = xt::zeros<T>({dim1, dim2});
  xt::xarray<T> data_res = xt::zeros<T>({dim1, dim2});

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_(i, j) = 1.0;
      data2_(i, j) = 2.0;
      data3_(i, j) = 3.0;
      data4_(i, j) = 4.0;
      data_res(i, j) = 0.0;
    }
  }

  {
    TimerRecorder a("xarray");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data1_ + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0);
}

template <class T>
void test_xtensor() {
  xt::xtensor<T, 2> data1_ = xt::zeros<T>({dim1, dim2});
  xt::xtensor<T, 2> data2_ = xt::zeros<T>({dim1, dim2});
  xt::xtensor<T, 2> data3_ = xt::zeros<T>({dim1, dim2});
  xt::xtensor<T, 2> data4_ = xt::zeros<T>({dim1, dim2});
  xt::xtensor<T, 2> data_res = xt::zeros<T>({dim1, dim2});

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      data1_(i, j) = 1.0;
      data2_(i, j) = 2.0;
      data3_(i, j) = 3.0;
      data4_(i, j) = 4.0;
      data_res(i, j) = 0.0;
    }
  }

  {
    TimerRecorder a("xtensor");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data1_ + data1_ - data2_ * data3_ / data4_;
    }
  }
  val = data_res(0, 0);
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

    // 静态分派 mdarray 测试
    if (dim1 == 1 && dim2 == 4) {
      test_mdarray_expr<double, 1, 4>();
      test_mdinpvec_expr<double, 1, 4>();
    } else if (dim1 == 1 && dim2 == 10) {
      test_mdarray_expr<double, 1, 10>();
      test_mdinpvec_expr<double, 1, 10>();
    } else if (dim1 == 1 && dim2 == 50) {
      test_mdarray_expr<double, 1, 50>();
      test_mdinpvec_expr<double, 1, 50>();
    } else if (dim1 == 3 && dim2 == 70) {
      test_mdarray_expr<double, 3, 70>();
      test_mdinpvec_expr<double, 3, 70>();
    } else if (dim1 == 5 && dim2 == 100) {
      test_mdarray_expr<double, 5, 100>();
      test_mdinpvec_expr<double, 5, 100>();
    } else if (dim1 == 10 && dim2 == 100) {
      test_mdarray_expr<double, 10, 100>();
      test_mdinpvec_expr<double, 10, 100>();
    } else if (dim1 == 100 && dim2 == 100) {
      test_mdarray_expr<double, 100, 100>();
      test_mdinpvec_expr<double, 100, 100>();
    }
    test_mdvector_expr<double>();
    test_simd<double>();
    test_transform<double>();
    test_eigen_tensor();
    test_base_expr<double>();
    test_valarray<double>();
    test_norm<double>();
    test_vector<double>();
    test_xtensor<double>();
    test_xarray<double>();
  }
  TimerRecorder::SaveSpeedResult("2d_speed_result.csv");
  std::cout << "test complete" << std::endl;

  return 0;
}