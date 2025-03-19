#include "src/test/test_set.h"
#include "xtensor/xarray.hpp"

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
      data3_ = data1_ - data2_ - data4_;
    }

    if constexpr (do_mul) {
      data3_ = data1_ * data2_ * data4_;
    }

    if constexpr (do_div) {
      data3_ = data1_ / data2_ / data4_;
    }
  }
}

int main(int args, char* argv[]) {
  // double
  test_xarray<double>();

  return 0;
}