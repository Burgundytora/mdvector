
#include "include_md.h"

void test_mdspan_dynamic() {
  try {
    constexpr size_t Rank = 3;
    std::array<size_t, Rank> shape{3, 3, 4};
    md::vector<double, Rank> mdvector_(shape);
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
  print_mdspan(mdspan_);

  std::mdspan<int, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_strided_(
      arr.data(), std::layout_stride::mapping(std::dextents<size_t, 2>{2, 3}, std::array<size_t, 2>{2, 5}));
  print_mdspan(mdspan_strided_);
  std::println("mdspan_strided_ size: {}", mdspan_strided_.size());
}

void test_md_inplace_vector() {
  md::inplace_vector<double, 3, 100> inp_vec;
  inp_vec.set_shape(2, 5, 10);
  inp_vec.fill(0.0);
  inp_vec += 2.0;
  inp_vec.print();
}

void test_mdvector() {
  md::vector<double, 2> data1(4, 4);
  md::vector<double, 2> data2(4, 4);
  md::vector<double, 2> data3(4, 4);
  data1.fill(1.0);
  data2.fill(2.0);
  data3.fill(3.0);
  std::println("before = &&tensor_extr");
  data3 = data1 + data2;
  std::println("before = &tensor_extr");
  md::vector<double, 2> data_new = data1 + data2;
}

int main() {
  // test_mdspan_dynamic();
  // test_layout_strided();
  // test_md_inplace_vector();
  test_mdvector();
  return 0;
}