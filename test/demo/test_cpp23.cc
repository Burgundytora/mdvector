#include <array>
#include <format>
#include <functional>
#include <iostream>
#include <numeric>
#include <print>
#include <string>
#include <utility>
#include <vector>

#include "mdvector.h"

void test_mdspan_dynamic() {
  try {
    constexpr size_t Rank = 3;
    std::array<size_t, Rank> shape{3, 3, 4};
    mdvector<int, Rank> mdvector_(shape);
    mdvector_.print();
    std::println("[0,0,0]: {}", mdvector_.at(1, 1, 1));
  } catch (const std::runtime_error& e) {
    std::cout << "正确捕获异常: " << e.what() << std::endl;
  }
}

int main() {
  test_mdspan_dynamic();
  return 0;
}