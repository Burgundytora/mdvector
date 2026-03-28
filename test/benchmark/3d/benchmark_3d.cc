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
#include "include_md_all.h"

double val = 0.0;

template <class T>
T*** create_3d_array(int x, int y, int z) {
  T*** arr = (T***)malloc(x * sizeof(T**));
  for (int i = 0; i < x; i++) {
    arr[i] = (T**)malloc(y * sizeof(T*));
    for (int j = 0; j < y; j++) {
      arr[i][j] = (T*)malloc(z * sizeof(T));
    }
  }
  return arr;
}

template <class T>
void free_3d_array(T*** arr, int x, int y) {
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      free(arr[i][j]);
    }
    free(arr[i]);
  }
  free(arr);
}

template <class T>
void test_norm() {
  T*** data1_ = create_3d_array<T>(dim1, dim2, dim3);
  T*** data2_ = create_3d_array<T>(dim1, dim2, dim3);
  T*** data3_ = create_3d_array<T>(dim1, dim2, dim3);
  T*** data4_ = create_3d_array<T>(dim1, dim2, dim3);
  T*** data_res = create_3d_array<T>(dim1, dim2, dim3);

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      for (size_t k = 0; k < dim3; k++) {
        data1_[i][j][k] = 1.0;
        data2_[i][j][k] = 2.0;
        data3_[i][j][k] = 3.0;
        data4_[i][j][k] = 4.0;
        data_res[i][j][k] = 0.0;
      }
    }
  }

  {
    TimerRecorder a("** 3d");

    size_t k = 0;
    while (k++ < loop) {
      for (int i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          for (size_t k = 0; k < dim3; k++) {
            data_res[i][j][k] =
                data_res[i][j][k] + data1_[i][j][k] - data2_[i][j][k] * data3_[i][j][k] / data4_[i][j][k];
          }
        }
      }
    }
  }
  val = data_res[0][0][0];

  free_3d_array(data1_, dim1, dim2);
  free_3d_array(data2_, dim1, dim2);
  free_3d_array(data3_, dim1, dim2);
  free_3d_array(data4_, dim1, dim2);
}

template <class T>
void test_transform() {
  vector<T> data1_(dim1 * dim2 * dim3);
  vector<T> data2_(dim1 * dim2 * dim3);
  vector<T> data3_(dim1 * dim2 * dim3);
  vector<T> data4_(dim1 * dim2 * dim3);
  vector<T> data_res(dim1 * dim2 * dim3);

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      for (size_t k = 0; j < dim3; j++) {
        data1_[i * dim2 * dim3 + j * dim3 + k] = 1.0;
        data2_[i * dim2 * dim3 + j * dim3 + k] = 2.0;
        data3_[i * dim2 * dim3 + j * dim3 + k] = 3.0;
        data4_[i * dim2 * dim3 + j * dim3 + k] = 4.0;
        data_res[i * dim2 * dim3 + j * dim3 + k] = 0.0;
      }
    }
  }

  TimerRecorder a("transform");

  size_t k = 0;
  while (k++ < loop) {
    std::transform(std::execution::unseq, data_res.begin(), data_res.end(), data_res.begin(), data_res.begin(),
                   std::plus<>());
    std::transform(std::execution::unseq, data_res.begin(), data_res.end(), data2_.begin(), data_res.begin(),
                   std::minus<>());
    std::transform(std::execution::unseq, data_res.begin(), data_res.end(), data3_.begin(), data_res.begin(),
                   std::multiplies<>());
    std::transform(std::execution::unseq, data_res.begin(), data_res.end(), data4_.begin(), data_res.begin(),
                   std::divides<>());
  }
  val = data_res[0];
}

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
      md::simd_add<T, md::aligned_policy>(data_res, data1_, data_res, total_element);
      md::simd_sub<T, md::aligned_policy>(data_res, data2_, data_res, total_element);
      md::simd_mul<T, md::aligned_policy>(data_res, data3_, data_res, total_element);
      md::simd_div<T, md::aligned_policy>(data_res, data4_, data_res, total_element);
    }
  }

  val = data_res[0];

  allocator_.deallocate(data1_);
  allocator_.deallocate(data2_);
  allocator_.deallocate(data3_);
}

template <class T>
void test_mdvector_expr() {
  md::shape<3> test_shape = {dim1, dim2, dim3};
  md::vector<T, 3> data1_(test_shape);
  md::vector<T, 3> data2_(test_shape);
  md::vector<T, 3> data3_(test_shape);
  md::vector<T, 3> data4_(test_shape);
  md::vector<T, 3> data_res(test_shape);

  // 赋值
  data1_.fill(1.0);
  data2_.fill(2.0);
  data3_.fill(3.0);
  data4_.fill(4.0);
  data4_.fill(4.0);
  data_res.fill(0.0);

  {
    TimerRecorder a("mdvector");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data_res + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0, 0);
}

template <class T, size_t N1, size_t N2, size_t N3>
void test_mdarray_expr() {
  md::array<T, std::layout_right, N1, N2, N3> data1_;
  md::array<T, std::layout_right, N1, N2, N3> data2_;
  md::array<T, std::layout_right, N1, N2, N3> data3_;
  md::array<T, std::layout_right, N1, N2, N3> data4_;
  md::array<T, std::layout_right, N1, N2, N3> data_res;

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
      data_res = data_res + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0, 0);
}

void test_eigen_tensor() {
  Eigen::Tensor<double, 3> data1_(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});
  Eigen::Tensor<double, 3> data2_(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});
  Eigen::Tensor<double, 3> data3_(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});
  Eigen::Tensor<double, 3> data4_(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});
  Eigen::Tensor<double, 3> data_res(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});

  for (int i = 0; i < dim1; ++i) {
    for (int j = 0; j < dim2; ++j) {
      for (int k = 0; k < dim3; ++k) {
        data1_(i, j, k) = 1.0;
        data2_(i, j, k) = 2.0;
        data3_(i, j, k) = 3.0;
        data4_(i, j, k) = 4.0;
        data_res(i, j, k) = 0.0;
      }
    }
  }

  {
    TimerRecorder a("eigen");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data_res + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0, 0);
}

template <class T>
void test_xarray() {
  xt::xarray<T> data1_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xarray<T> data2_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xarray<T> data3_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xarray<T> data4_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xarray<T> data_res = xt::zeros<T>({dim1, dim2, dim3});

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      for (size_t k = 0; k < dim3; k++) {
        data1_(i, j, k) = 1.0;
        data2_(i, j, k) = 2.0;
        data3_(i, j, k) = 3.0;
        data4_(i, j, k) = 4.0;
        data_res(i, j, k) = 0.0;
      }
    }
  }

  {
    TimerRecorder a("xarray");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data_res + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0, 0);
}

template <class T>
void test_xtensor() {
  xt::xtensor<T, 3> data1_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xtensor<T, 3> data2_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xtensor<T, 3> data3_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xtensor<T, 3> data4_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xtensor<T, 3> data_res = xt::zeros<T>({dim1, dim2, dim3});

  // 赋值
  for (int i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      for (size_t k = 0; k < dim3; k++) {
        data1_(i, j, k) = 1.0;
        data2_(i, j, k) = 2.0;
        data3_(i, j, k) = 3.0;
        data4_(i, j, k) = 4.0;
        data_res(i, j, k) = 0.0;
      }
    }
  }

  {
    TimerRecorder a("xtensor");

    size_t k = 0;
    while (k++ < loop) {
      data_res = data_res + data1_ - data2_ * data3_ / data4_;
    }
  }

  val = data_res(0, 0, 0);
}

int main(int args, char* argv[]) {
  md::print_simd_type();

  try {
    for (const auto& test : all_test_points) {
      loop = test.loop_;
      dim1 = test.dim1_;
      dim2 = test.dim2_;
      dim3 = test.dim3_;
      total_element = test.total_element_;
      total_cal = test.total_cal_;

      std::cout << "3d: " << dim1 << "*" << dim2 << "*" << dim3 << "\n";

      // // double

      // test_mdarray_expr<double, dim1, dim2, dim3>();
      // 静态分派 mdarray 测试
      if (dim1 == 2 && dim2 == 2 && dim3 == 2) {
        test_mdarray_expr<double, 2, 2, 2>();
      } else if (dim1 == 3 && dim2 == 3 && dim3 == 3) {
        test_mdarray_expr<double, 3, 3, 3>();
      } else if (dim1 == 5 && dim2 == 5 && dim3 == 5) {
        test_mdarray_expr<double, 5, 5, 5>();
      } else if (dim1 == 7 && dim2 == 7 && dim3 == 7) {
        test_mdarray_expr<double, 7, 7, 7>();
      } else if (dim1 == 10 && dim2 == 10 && dim3 == 10) {
        test_mdarray_expr<double, 10, 10, 10>();
      } else if (dim1 == 20 && dim2 == 20 && dim3 == 20) {
        test_mdarray_expr<double, 20, 20, 20>();
      } else if (dim1 == 30 && dim2 == 30 && dim3 == 30) {
        test_mdarray_expr<double, 30, 30, 30>();
      }
      test_transform<double>();
      test_simd<double>();
      test_mdvector_expr<double>();
      test_eigen_tensor();
      test_norm<double>();
      test_xtensor<double>();
      test_xarray<double>();
    }
  } catch (const std::exception& e) {
    std::cout << "error: " << e.what();
  }

  TimerRecorder::SaveSpeedResult("3d_speed_result.csv");
  std::cout << "test complete" << std::endl;

  return 0;
}