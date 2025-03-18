#include "src/avx2/mdvector.h"
#include "test_set.h"

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
  std::cout << "2d matrix ? matrix: " << dim1 << "*" << dim2 << "\n";

  // double
  test_mdvector_expr<double>();

  std::cout << "test complete" << std::endl;

  return 0;
}
