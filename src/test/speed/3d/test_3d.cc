#include "test_set.h"
#include "time_cost.h"

//

#include "src/mdvector/mdvector.h"
#include "src/mdvector/simd/simd_function.h"

//
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

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

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      for (size_t k = 0; k < dim3; k++) {
        data1_[i][j][k] = 1;
        data2_[i][j][k] = 2;
        data4_[i][j][k] = 3;
      }
    }
  }

  TimerRecorder a("** 3d");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          for (size_t k = 0; k < dim3; k++) {
            data3_[i][j][k] = data1_[i][j][k] + data2_[i][j][k];
          }
        }
      }
    }

    if constexpr (do_sub) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          for (size_t k = 0; k < dim3; k++) {
            data3_[i][j][k] = data1_[i][j][k] - data2_[i][j][k];
          }
        }
      }
    }

    if constexpr (do_mul) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          for (size_t k = 0; k < dim3; k++) {
            data3_[i][j][k] = data1_[i][j][k] * data2_[i][j][k];
          }
        }
      }
    }

    if constexpr (do_div) {
      for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
          for (size_t k = 0; k < dim3; k++) {
            data3_[i][j][k] = data1_[i][j][k] / data2_[i][j][k];
          }
        }
      }
    }
  }

  free_3d_array(data1_, dim1, dim2);
  free_3d_array(data2_, dim1, dim2);
  free_3d_array(data3_, dim1, dim2);
  free_3d_array(data4_, dim1, dim2);
}

template <class T>
void test_simd() {
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

  TimerRecorder a("simd 1d");

  size_t k = 0;
  while (k++ < loop) {
    if constexpr (do_add) {
      simd_add<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_sub) {
      simd_sub<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_mul) {
      simd_mul<T>(data1_, data2_, data3_, total_element);
    }

    if constexpr (do_div) {
      simd_div<T>(data1_, data2_, data3_, total_element);
    }
  }

  allocator_.deallocate(data1_);
  allocator_.deallocate(data2_);
  allocator_.deallocate(data3_);
}

template <class T>
void test_mdvector_expr() {
  mdshape_3d test_shape = {dim1, dim2, dim3};
  mdvector_3d<T> data1_(test_shape);
  mdvector_3d<T> data2_(test_shape);
  mdvector_3d<T> data3_(test_shape);
  mdvector_3d<T> data4_(test_shape);

  // 赋值
  data1_.set_value(1);
  data2_.set_value(2);
  data4_.set_value(3);

  TimerRecorder a("md expr");

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
  }
}

void test_eigen() {
  Eigen::Tensor<double, 3> data1_(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});
  Eigen::Tensor<double, 3> data2_(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});
  Eigen::Tensor<double, 3> data3_(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});
  Eigen::Tensor<double, 3> data4_(Eigen::array<Eigen::Index, 3>{
      static_cast<Eigen::Index>(dim1), static_cast<Eigen::Index>(dim2), static_cast<Eigen::Index>(dim3)});

  for (int i = 0; i < dim1; ++i) {
    for (int j = 0; j < dim2; ++j) {
      for (int k = 0; k < dim3; ++k) {
        data1_(i, j, k) = 1;  // 示例值
        data2_(i, j, k) = 2;  // 示例值
        data4_(i, j, k) = 3;  // 示例值
      }
    }
  }

  TimerRecorder a("eigen");

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
  }
}

template <class T>
void test_xarray() {
  xt::xarray<T> data1_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xarray<T> data2_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xarray<T> data3_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xarray<T> data4_ = xt::zeros<T>({dim1, dim2, dim3});

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      for (size_t k = 0; k < dim3; k++) {
        data1_(i, j, k) = 1;
        data2_(i, j, k) = 2;
        data4_(i, j, k) = 3;
      }
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
  }
}

template <class T>
void test_xtensor() {
  xt::xtensor<T, 3> data1_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xtensor<T, 3> data2_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xtensor<T, 3> data3_ = xt::zeros<T>({dim1, dim2, dim3});
  xt::xtensor<T, 3> data4_ = xt::zeros<T>({dim1, dim2, dim3});

  // 赋值
  for (size_t i = 0; i < dim1; i++) {
    for (size_t j = 0; j < dim2; j++) {
      for (size_t k = 0; k < dim3; k++) {
        data1_(i, j, k) = 1;
        data2_(i, j, k) = 2;
        data4_(i, j, k) = 3;
      }
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
  }
}

void print_simd_type() {
#if defined(USE_AVX2)
  std::cout << "avx2...\n";

#elif defined(USE_AVX512)
  std::cout << "avx512...\n";

#elif defined(USE_SSE)
  std::cout << "sse...\n";

#elif defined(USE_NEON)
  std::cout << "neon...\n";

#elif defined(USE_RVV)
  std::cout << "risc_v...\n";

#else
  std::cout << "default avx2...\n";
#endif
}

int main(int args, char* argv[]) {
  print_simd_type();

  for (const auto& test : all_test_points) {
    loop = test.loop_;
    dim1 = test.dim1_;
    dim2 = test.dim2_;
    dim3 = test.dim3_;
    total_element = test.total_element_;
    total_cal = test.total_cal_;

    std::cout << "3d: " << dim1 << "*" << dim2 << "*" << dim3 << "\n";

    // // double
    test_simd<double>();
    test_mdvector_expr<double>();
    test_eigen();

    test_norm<double>();
    test_xtensor<double>();
    test_xarray<double>();
  }
  TimerRecorder::SaveSpeedResult("3d_speed_result.csv");
  std::cout << "test complete" << std::endl;

  return 0;
}