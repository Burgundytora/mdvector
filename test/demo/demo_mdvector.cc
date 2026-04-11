#include "include_md_all.h"
#include <iostream>
#include <iomanip>
#include <chrono>

// 辅助打印函数
template <typename T, size_t Rank>
void print_vector(const md::vector<T, Rank>& vec, const std::string& name) {
  std::cout << "\n=== " << name << " ===" << std::endl;
  std::cout << "Shape: [";
  auto extents = vec.extents();
  for (size_t i = 0; i < Rank; ++i) {
    std::cout << extents[i];
    if (i < Rank - 1) std::cout << " x ";
  }
  std::cout << "], Size: " << vec.size() << std::endl;
  vec.print();
}

template <typename T>
void print_scalar(const T& value, const std::string& name) {
  std::cout << "\n=== " << name << " ===" << std::endl;
  std::cout << value << std::endl;
}

// mdvector功能示例
int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "md::vector Complete Demo" << std::endl;
  std::cout << "========================================" << std::endl;

  // ========== 1. 构造函数和初始化 ==========
  std::cout << "\n========== 1. Constructor and Initialization ==========" << std::endl;

  // 1.1 基本构造
  vector_1d<int> v1_5(5);  // 1维，5个元素
  v1_5.set_arange(0, 1);   // [0,1,2,3,4]
  print_vector(v1_5, "v1_5 (1D with 5 elements)");

  vector_2d<float> v2_3x4(3, 4);  // 2维，3x4
  v2_3x4.set_arange(1.0f, 2.0f);  // 线性填充
  print_vector(v2_3x4, "v2_3x4 (2D 3x4 with linear fill)");

  vector_3d<double> v3_2x3x4(2, 3, 4);  // 3维，2x3x4
  v3_2x3x4.fill(3.14);                  // 填充固定值
  print_vector(v3_2x3x4, "v3_2x3x4 (3D 2x3x4 filled with 3.14)");

  // 1.2 使用初始化列表构造 暂不支持
  // vector_1d<int> v1_init = {1, 2, 3, 4, 5};

  // 1.3 拷贝构造和移动构造
  auto v1_copy = v1_5;
  print_vector(v1_copy, "v1_copy (copy of v1_5)");

  auto v1_move = std::move(v1_copy);
  print_vector(v1_move, "v1_move (move from v1_copy)");

  // ========== 2. 填充函数 ==========
  std::cout << "\n========== 2. Fill Functions ==========" << std::endl;

  vector_1d<int> fill_demo(8);

  fill_demo.set_arange();
  print_vector(fill_demo, "arange() - [0,1,2,3,4,5,6,7]");

  fill_demo.set_arange(10, 2);
  print_vector(fill_demo, "arange(10, 2) - [10,12,14,16,18,20,22,24]");

  fill_demo.set_arange(0, 3);
  print_vector(fill_demo, "arange(0, 3) - [0, 3, 6, ..., 21]");

  fill_demo.set_zeros();
  print_vector(fill_demo, "zeros() - all zeros");

  fill_demo.set_ones();
  print_vector(fill_demo, "ones() - all ones");

  fill_demo.fill(42);
  print_vector(fill_demo, "fill(42) - all 42");

  // ========== 3. 多维索引访问 ==========
  std::cout << "\n========== 3. Multi-dimensional Indexing ==========" << std::endl;

  vector_2d<int> idx_demo(3, 3);
  idx_demo.set_arange(1, 1);  // [1,2,3,4,5,6,7,8,9]
  print_vector(idx_demo, "idx_demo (3x3 matrix)");

  // 使用 operator() 访问
  std::cout << "Element (0,0): " << idx_demo(0, 0) << std::endl;
  std::cout << "Element [1,1]: " << idx_demo[1, 1] << std::endl;
  std::cout << "Element .at(2,2): " << idx_demo.at(2, 2) << std::endl;

  // 修改元素
  idx_demo[1, 1] = 100;
  std::cout << "After modifying [1,1] to 100:" << std::endl;
  idx_demo.print();

  // 使用 at() 进行边界检查
  try {
    std::cout << "Trying to access .at(3,0): ";
    std::cout << idx_demo.at(3, 0) << std::endl;
  } catch (const std::out_of_range& e) {
    std::cout << "Caught exception: " << e.what() << std::endl;
  }

  // ========== 4. 索引转换 ==========
  std::cout << "\n========== 4. Index Conversion ==========" << std::endl;

  vector_2d<int> idx_conv(2, 3);
  idx_conv.set_arange(0, 1);
  print_vector(idx_conv, "idx_conv (2x3 matrix)");

  // 多维索引转一维索引
  size_t linear = idx_conv.get_1d_index(1, 2);
  std::cout << "get_1d_index(1, 2) -> " << linear << std::endl;

  // 一维索引转多维索引
  auto multi_idx = idx_conv.get_md_index(5);
  std::cout << "get_md_index(5) -> (" << multi_idx[0] << ", " << multi_idx[1] << ")" << std::endl;

  // 获取指定维度的索引
  size_t dim0 = idx_conv.get_dim_index(5, 0);
  size_t dim1 = idx_conv.get_dim_index(5, 1);
  std::cout << "get_dim_index(5, 0) -> " << dim0 << ", get_dim_index(4, 1) -> " << dim1 << std::endl;

  // ========== 5. 表达式模板和数值计算 ==========
  std::cout << "\n========== 5. Expression Template and Arithmetic ==========" << std::endl;

  vector_1d<double> a(5);
  vector_1d<double> b(5);
  a.set_arange(1, 1);   // [1,2,3,4,5]
  b.set_arange(5, -1);  // [5,4,3,2,1]

  print_vector(a, "a = [1,2,3,4,5]");
  print_vector(b, "b = [5,4,3,2,1]");

  // 向量加法
  vector_1d<double> c = a + b;
  print_vector(c, "c = a + b");

  // 向量减法
  vector_1d<double> d = a - b;
  print_vector(d, "d = a - b");

  // 向量乘法
  vector_1d<double> e = a * b;
  print_vector(e, "e = a * b");

  // 向量除法
  vector_1d<double> f = a / b;
  f.print();
  print_vector(f, "f = a / b");

  // 标量运算
  vector_1d<double> g = a + 10.0;
  print_vector(g, "g = a + 10");

  vector_1d<double> h = a * 2.0;
  print_vector(h, "h = a * 2");

  // 复合赋值
  vector_1d<double> comp(5);
  comp.set_arange(1, 1);
  comp += 5.0;
  print_vector(comp, "comp += 5");

  comp *= 2.0;
  print_vector(comp, "comp *= 2");

  // 取负
  vector_1d<double> neg = -a;
  print_vector(neg, "neg = -a");

  // 复杂表达式
  vector_1d<double> complex_expr = (a + b) * 2.0 - (a / b) + 10.0;
  print_vector(complex_expr, "complex_expr = (a + b) * 2 - (a / b) + 10");

  // ========== 6. 视图操作 (span) ==========
  std::cout << "\n========== 6. Span Operations (Contiguous Views) ==========" << std::endl;

  vector_2d<int> matrix(4, 4);
  matrix.set_arange(1, 1);  // 1-16
  print_vector(matrix, "Matrix (4x4)");

  // 获取行视图（连续内存）
  auto row_span = matrix.span(2, all());  // 第3行
  std::cout << "Row 2 (0-indexed) as span: ";
  for (size_t i = 0; i < row_span.size(); ++i) {
    std::cout << row_span[i] << " ";
  }
  std::cout << std::endl;

  // 获取列视图（注意：列在行优先布局中不连续）
  try {
    auto col_span = matrix.span(all(), 2);  // 第3列
    std::cout << "Column 2 as span: ";
    for (size_t i = 0; i < col_span.size(); ++i) {
      std::cout << col_span[i] << " ";
    }
    std::cout << std::endl;
  } catch (const std::runtime_error& e) {
    std::cout << "Column span not contiguous: " << e.what() << std::endl;
  }

  // 获取子矩阵（连续区域）
  try {
    auto sub_span = matrix.span(1, slice(1, 3));
    std::cout << "Submatrix (rows 1, cols 1-3):" << std::endl;
    sub_span.print();
  } catch (const std::runtime_error& e) {
    std::cout << "Column span not contiguous: " << e.what() << std::endl;
  }

  // ========== 7. 视图操作 (view with stride) ==========
  std::cout << "\n========== 7. View Operations (Strided Views) ==========" << std::endl;

  vector_2d<int> strided_demo(4, 4);
  strided_demo.set_arange(1, 1);
  print_vector(strided_demo, "Original matrix");

  // 创建步长视图（每隔一个元素取一个）
  auto strided_view = strided_demo.view(all(), slice(0, 2, 3));  // 每隔一列
  std::cout << "Strided view (rows all, cols 1 and 3):" << std::endl;
  strided_view.print();

  // 修改视图中的值会影响原矩阵
  strided_view[0, 0] = 999;
  std::cout << "\nAfter modifying view [0,0] to 999:" << std::endl;
  std::cout << "Original matrix at [0,0]: " << strided_demo[0, 0] << std::endl;
  std::cout << "Original matrix at [0,1]: " << strided_demo[0, 1] << std::endl;

  // ========== 8. 维度操作 ==========
  std::cout << "\n========== 8. Dimension Operations ==========" << std::endl;

  vector_2d<int> reshape_demo(2, 3);
  reshape_demo.set_arange(1, 1);
  print_vector(reshape_demo, "Original (2x3)");

  // 改变形状（元素总数不变）
  reshape_demo.set_shape(3, 2);
  print_vector(reshape_demo, "Reshaped to (3x2)");

  std::cout << "Rank: " << reshape_demo.rank() << std::endl;
  std::cout << "Extent of dimension 0: " << reshape_demo.extent(0) << std::endl;
  std::cout << "Extent of dimension 1: " << reshape_demo.extent(1) << std::endl;

  // ========== 9. 迭代器使用 ==========
  std::cout << "\n========== 9. Iterator Usage ==========" << std::endl;

  vector_1d<int> iter_demo(10);
  iter_demo.set_arange(0, 2);
  print_vector(iter_demo, "iter_demo");

  // 使用范围for循环
  std::cout << "Range-based for loop: ";
  for (const auto& val : iter_demo) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  // 使用迭代器
  std::cout << "Iterator (begin to end): ";
  for (auto it = iter_demo.begin(); it != iter_demo.end(); ++it) {
    std::cout << *it << " ";
  }
  std::cout << std::endl;

  // 反向迭代器
  std::cout << "Reverse iterator: ";
  for (auto it = iter_demo.rbegin(); it != iter_demo.rend(); ++it) {
    std::cout << *it << " ";
  }
  std::cout << std::endl;

  // 使用算法
  int sum = std::accumulate(iter_demo.begin(), iter_demo.end(), 0);
  std::cout << "Sum of all elements: " << sum << std::endl;

  auto max_it = std::max_element(iter_demo.begin(), iter_demo.end());
  std::cout << "Maximum element: " << *max_it << std::endl;

  // ========== 10. 错误处理和边界检查 ==========
  std::cout << "\n========== 10. Error Handling and Bounds Checking ==========" << std::endl;

  vector_2d<int> error_demo(2, 3);

  // 测试越界访问
  try {
    error_demo.at(5, 5) = 10;
  } catch (const std::out_of_range& e) {
    std::cout << "Out of range caught: " << e.what() << std::endl;
  }

  // 测试未初始化访问
  vector_2d<int> uninit_demo;
  try {
    uninit_demo.at(0, 0) = 10;
  } catch (const std::logic_error& e) {
    std::cout << "Uninitialized access caught: " << e.what() << std::endl;
  }

  // 测试形状不匹配
  try {
    vector_2d<int> shape_error(2, 3);
    shape_error.set_shape(2, 4);  // 元素总数从6变为8
    std::cout << "Shape changed successfilly, new size: " << shape_error.size() << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Shape change error: " << e.what() << std::endl;
  }

  // ========== 11. 性能测试 ==========
  std::cout << "\n========== 11. Performance Demo ==========" << std::endl;

  const int large_size = 10;
  vector_1d<double> large_vec1(large_size);
  vector_1d<double> large_vec2(large_size);
  vector_1d<double> result_seq(large_size);
  vector_1d<double> result_par(large_size);

  // large_vec1.set_arange(0.0, 1.0);
  // large_vec2.set_arange(large_size, -1.0);
  large_vec1.fill(1.0);
  large_vec2.fill(2.0);

  auto expr = large_vec1 + large_vec2 * large_vec1 - 2.0 * large_vec2 + large_vec2 / large_vec1;
  auto expr2 = large_vec1 + large_vec2;

  auto start = std::chrono::high_resolution_clock::now();
  result_seq = expr2;
  auto end = std::chrono::high_resolution_clock::now();
  result_seq.print();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  auto start_par = std::chrono::high_resolution_clock::now();
  expr2.eval_to(result_par, md::par);
  auto end_par = std::chrono::high_resolution_clock::now();
  result_par.print();
  auto duration_par = std::chrono::duration_cast<std::chrono::microseconds>(end_par - start_par);

  auto num_commas = [](int value) {
    std::string str = std::to_string(value);
    std::string result;

    int count = 0;
    // 从后往前遍历，每3位加一个逗号
    for (auto it = str.rbegin(); it != str.rend(); ++it) {
      if (count > 0 && count % 3 == 0) {
        result.push_back(',');
      }
      result.push_back(*it);
      count++;
    }

    std::reverse(result.begin(), result.end());
    return result;
  };

  std::cout << "Vector addition of " << num_commas(large_size)
            << " elements took: " << static_cast<double>(duration.count()) / 1000.0 << " ms"
            << "   parallel tool: " << static_cast<double>(duration_par.count()) / 1000.0 << " ms" << std::endl;

  // ========== 12. 复杂示例 ==========
  std::cout << "\n========== 12. Complex Using Demo ==========" << std::endl;

  // 快速根据节点三维坐标计算10个梁的长度
  md::vector<double, 2> pos_info(3, 11);     //      node1 node2 node3 ... node11
  pos_info.span(0, all()).set_arange(1, 1);  // x坐标   1    2     3   ...   11
  pos_info.span(1, all()).set_arange(2, 2);  // y坐标   2    4     6   ...   22
  pos_info.span(2, all()).set_arange(3, 3);  // z坐标   3    6     9   ...   33
  std::cout << "node position:" << std::endl;
  pos_info.print();
  auto x1 = pos_info.span(0, slice(0, -2));
  auto y1 = pos_info.span(1, slice(0, -2));
  auto z1 = pos_info.span(2, slice(0, -2));
  auto x2 = pos_info.span(0, slice(1, -1));
  auto y2 = pos_info.span(1, slice(1, -1));
  auto z2 = pos_info.span(2, slice(1, -1));
  auto length = hypot(x2 - x1, y2 - y1, z2 - z1);
  std::cout << "element length:\n";
  length.print();

  std::cout << "\n========================================" << std::endl;
  std::cout << "md::vector Demo Completed Successfilly!" << std::endl;
  std::cout << "========================================" << std::endl;
  print_simd_type();

  return 0;
}