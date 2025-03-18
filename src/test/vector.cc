#include <vector>

#include "test_set.h"

using std::vector;

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
          data3_[i][j] = data1_[i][j] + data2_[i][j];
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

int main(int args, char* argv[]) {
  std::cout << "2d matrix ? matrix: " << dim1 << "*" << dim2 << "\n";

  // double
  test_vector<double>();

  std::cout << "test complete" << std::endl;

  return 0;
}