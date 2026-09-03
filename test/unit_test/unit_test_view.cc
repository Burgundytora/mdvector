
#include "include_md_all.h"

int main() {
  try {
    std::cout << "=== 测试1: 10x10矩阵基本操作 ===" << std::endl;
    md::vector<double, 2> mat(10, 10);

    // 填充数据
    mat.set_arange();

    std::println("mat:");
    mat.print();

    std::println("mat: 1d_index of [1,1]:{}", mat.get_1d_index(1, 1));

    std::println("mat: md_index of 15:{}", mat.get_md_index(15));

    std::println("view(slice(1, 2, -1), slice(1, 2, -1)):");
    auto view_1 = mat.view(slice(1, 2, -1), slice(1, 2, -1));
    view_1.print();

    std::println("view[1,1]:{}", view_1[1, 1]);

    std::println("view: 1d_index of [2,2]:{}", view_1.get_1d_index(2, 2));

    std::println("view: md_index of 13:{}", view_1.get_md_index(13));

    std::println("iterator of view:");
    int index = 0;
    for (const auto& it : view_1) {
      std::println("i:{}, v:{}", index++, it);
    }

    md::vector<double, 2> mat2(5, 5);
    mat2.fill(2);

    view_1 *= 1.0;
    md::vector<double, 2> mat_res = view_1 * 2.0 + mat2 + cos(view_1);
    std::println("view*2 + mat2 + cos(-view):");
    mat_res.print();

    view_1 = view_1 * -mat2;
    std::println("view_1 * -mat2:");
    view_1.print();

    // Reverse and empty slices use signed strides and remain bounds-safe.
    auto reversed = mat.view(slice(9, -1, 0), all());
    if (reversed.extents() != std::array<size_t, 2>{10, 10} || reversed(0, 0) != mat(9, 0) ||
        reversed(9, 9) != mat(0, 9)) return 1;
    auto empty = mat.view(slice(4, 1), all());
    if (empty.extent(0) != 0 || empty.size() != 0) return 1;
    const auto& const_mat = mat;
    auto read_only = const_mat.view(slice(0, 2), all());
    static_assert(std::is_const_v<typename decltype(read_only)::value_type>);
    if (read_only(0, 0) != mat(0, 0)) return 1;

  } catch (const std::exception& e) {
    std::cout << "error: " << e.what() << std::endl;
  }

  std::cout << "view test down. \n";

  return 0;
}
