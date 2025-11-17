
#include "include_md.h"

int main(int args, char *argv[]) {
  md::shape<3> shape({2, 3, 4});
  md::vector<double, 3> test_vector3d(shape);
  md::vector<double, 3, std::layout_left> test_vector3d_layout_left(shape);
  std::cout << "mdvector 3d: shape 2 3 4 :\n";

  double t = 1.0;
  for (auto &it : test_vector3d) {
    it = t++;
  }
  t = 1.0;
  for (auto &it : test_vector3d_layout_left) {
    it = t++;
  }

  std::cout << "mdvector: print layout right ijk 1~24 :\n";
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      for (int k = 0; k < shape[2]; k++) {
        std::cout << i << " " << j << " " << k << ": " << test_vector3d(i, j, k) << "\n";
      }
    }
  }
  std::cout << "\n";

  std::cout << "mdvector: print layout left ijk 1~24 :\n";
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      for (int k = 0; k < shape[2]; k++) {
        std::cout << i << " " << j << " " << k << ": " << test_vector3d_layout_left(i, j, k) << "\n";
      }
    }
  }
  std::cout << "\n";

  std::cout << "mdvector: print layout right kji 1~24 :\n";
  for (int k = 0; k < shape[2]; k++) {
    for (int j = 0; j < shape[1]; j++) {
      for (int i = 0; i < shape[0]; i++) {
        std::cout << i << " " << j << " " << k << ": " << test_vector3d(i, j, k) << "\n";
      }
    }
  }
  std::cout << "\n";

  std::cout << "mdvector: print layout left kji 1~24 :\n";
  for (int k = 0; k < shape[2]; k++) {
    for (int j = 0; j < shape[1]; j++) {
      for (int i = 0; i < shape[0]; i++) {
        std::cout << i << " " << j << " " << k << ": " << test_vector3d_layout_left(i, j, k) << "\n";
      }
    }
  }
  std::cout << "\n";

  // ok
  md::span<double, 1> layout_right_span_1 = test_vector3d.span(0, 1, all());
  std::cout << "layout_right_span_1: \n";
  for (const auto &it : layout_right_span_1) {
    std::cout << it << " ";
  }

  // wrong
  try {
    md::span<double, 1, std::layout_left> layout_left_span_1 = test_vector3d_layout_left.span(0, 1, all());
  } catch (const std::exception &e) {
    std::cout << "\n捕获异常: layout_left_span_1 " << e.what() << std::endl;
  }

  // wrong
  try {
    md::span<double, 1> layout_right_span_2 = test_vector3d.span(all(), 0, 0);
  } catch (const std::exception &e) {
    std::cout << "\n捕获异常: layout_right_span_2 " << e.what() << std::endl;
  }

  // right
  md::span<double, 1, std::layout_left> layout_left_span_2 = test_vector3d_layout_left.span(all(), 1, 2);
  std::cout << "layout_left_span_2: \n";
  for (const auto &it : layout_left_span_2) {
    std::cout << it << " ";
  }
  std::cout << "\n";

  std::cout << "layout test down. \n";

  return 0;
}