
#include "include_md.h"

int main() {
  std::cout << "=== 测试1: 4x4x4矩阵基本操作 ===" << std::endl;
  mdvector<double, 2> mat({8, 8});

  // 填充数据
  for (int i = 0; i < mat.extent(0); ++i) {
    for (int j = 0; j < mat.extent(1); ++j) {
      mat(i, j) = j + i * 10;
    }
  }

  std::println("mat:");
  mat.print();

  std::println("span(1, all()):");
  auto span_1 = mat.span(1, slice(0, 5));
  span_1.print();

  std::println("view(2, slice(1, 3, 7)):");
  auto view_1 = mat.view(slice(1, 2, 5), slice(1, 2, 3));
  view_1.print();
  // std::println("view_1:{}", *view_1);

  return 0;
}