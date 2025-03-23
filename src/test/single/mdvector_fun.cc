#include "src/mdvector/mdvector.h"
#include "src/test/test_set.h"

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
      // data3_.equal_a_add_b(data3_, data4_);
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

int main(int args, char* argv[]) {
  // double
  test_mdvector_fun<double>();

  return 0;
}