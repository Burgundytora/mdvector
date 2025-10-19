
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

void test_layout_strided() {
  std::array<int, 16> arr;
  std::iota(arr.begin(), arr.end(), 1);
  std::println("arr: {}", arr);
  std::mdspan<int, std::extents<std::size_t, 4, 4>, std::layout_right> mdspan_(arr.data());
  md::print_mdspan(mdspan_);

  std::mdspan<int, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_strided_(
      arr.data(), std::layout_stride::mapping(std::dextents<size_t, 2>{4, 2}, std::array<size_t, 2>{4, 2}));
  md::print_mdspan(mdspan_strided_);
  std::println("mdspan_strided_ size: {}", mdspan_strided_.size());
}

int main() {
  // test_mdspan_dynamic();
  test_layout_strided();
  return 0;
}