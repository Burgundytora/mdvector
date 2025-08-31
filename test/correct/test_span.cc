#include <iostream>

#include "mdvector.h"

#if defined(_WIN32)
#include <windows.h>  // 添加Windows头文件
#endif

using md::all;
using md::slice;
using md::span;

int main() {
#if defined(_WIN32)
  // 设置控制台输出为UTF-8编码
  SetConsoleOutputCP(65001);
#endif

  // 测试1: 创建3x3矩阵并填充数据
  std::cout << "=== 测试1: 3x3矩阵基本操作 ===" << std::endl;
  mdvector<double, 2> mat({3, 3});

  // 填充数据
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      mat(i, j) = i * 3 + j + 1;  // 1-9
    }
  }

  std::cout << "原始矩阵:" << std::endl;
  mat.show_data_matrix_style();

  // 测试2: 创建子视图

  // 情况1: 最后一维完整切片
  span<double, 1> sub_contig1 = mat.span(1,     // 选择第1行
                                         all()  // 所有列
  );

  auto sub_contig_test = mat.span(2,     // 选择第1行
                                  all()  // 所有列
  );

  std::cout << "\n子视图(行2, 所有列):" << std::endl;
  for (int i = 0; i < sub_contig1.extent(0); ++i) {
    std::cout << sub_contig1(i) << " ";
  }
  std::cout << std::endl;
  // 预期输出:
  // 4 5 6

  // 情况1: 最后一维完整切片
  auto sub_contig11 = mat.span(1,            // 选择第1行
                               slice(1, -1)  // 第一列之后
  );

  std::cout << "\n子视图(行2, 列1:-1):" << std::endl;

  for (int i = 0; i < sub_contig11.extent(0); ++i) {
    std::cout << sub_contig11(i) << " ";
  }
  std::cout << std::endl;
  // 预期输出:
  // 5 6

  // 情况1: 最后一维完整切片
  auto sub_contig111 = mat.span(-1,    // 选择第1行
                                all()  // 第一列之后
  );

  std::cout << "\n子视图(行-1, 列:):" << std::endl;
  for (int i = 0; i < sub_contig111.extent(0); ++i) {
    std::cout << sub_contig111(i) << " ";
  }
  std::cout << std::endl;
  // 预期输出:
  // 7 8 9

  // 情况2: 单行选择
  auto sub_contig2 = mat.span(1,           // 只选择第2行
                              slice(0, 1)  // 选择第1-2列
  );

  std::cout << "\n子视图(行1, 列0:1):" << std::endl;
  for (int j = 0; j < sub_contig2.extent(0); ++j) {
    std::cout << sub_contig2(j) << " ";
  }
  // 预期输出: 4 5

  // 情况3: 非法情况测试
  std::cout << "\n\n=== 测试非法子视图 ===" << std::endl;
  try {
    auto invalid_sub = mat.span(slice(0, 2, false),  // 多行
                                slice(0, 1, false)   // 多列
    );
    std::cout << "错误：非法子视图创建成功！" << std::endl;
  } catch (const std::runtime_error& e) {
    std::cout << "正确捕获异常: " << e.what() << std::endl;
  }
  // 预期输出: 正确捕获异常: span slices must result in contiguous memory

  // 测试3: 使用语法糖创建子视图
  std::cout << "\n=== 测试3: 使用语法糖 ===" << std::endl;
  auto sub2 = mat.span(slice(0, 1),  // 第0-1行
                       all()         // 所有列
  );

  std::cout << "子视图(0:1, :):" << std::endl;
  for (int i = 0; i < sub2.extent(0); ++i) {
    for (int j = 0; j < sub2.extent(1); ++j) {
      std::cout << sub2(i, j) << " ";
    }
    std::cout << std::endl;
  }
  // 预期输出:
  // 1 2 3
  // 4 5 6

  // 测试4: 修改子视图影响原数据
  std::cout << "\n=== 测试4: 通过子视图修改数据 ===" << std::endl;
  sub_contig1(0) = 99;  // 修改子视图的第一个元素

  std::cout << "修改后的原始矩阵:" << std::endl;
  mat.show_data_matrix_style();
  // 预期输出中mat[1][0]变为99

  // 表达式计算
  std::cout << "\n=== 测试5: 子视图元素*10 ===" << std::endl;
  sub_contig1 *= 10.0;
  std::cout << "修改后的原始矩阵:" << std::endl;
  mat.show_data_matrix_style();
  // 第二行 990 50 60

  std::cout << "\n=== 测试5: 子视图元素+10 ===" << std::endl;
  sub_contig1 = sub_contig1 + 10.0;
  std::cout << "修改后的原始矩阵:" << std::endl;
  mat.show_data_matrix_style();
  // 第二行 1000 60 70

  std::cout << "\n=== 测试5: 子视图元素 = 第三行/0.5 ===" << std::endl;
  sub_contig1 = sub_contig_test / 0.5;
  std::cout << "修改后的原始矩阵:" << std::endl;
  mat.show_data_matrix_style();
  // 第二行 14 16 18
  sub_contig1.show_data_matrix_style();
  // 14 16 18

  // // 测试5: 尝试创建非法子视图
  try {
    auto invalid_sub = mat.span(slice(2, 4),  // 超出范围
                                slice(0, 3));
  } catch (const std::exception& e) {
    std::cout << "\n捕获异常: " << e.what() << std::endl;
  }

  // 测试6: 3D数组子视图
  std::cout << "\n=== 测试6: 3D数组测试 ===" << std::endl;
  mdvector<double, 3> tensor({2, 3, 4});

  // 填充3D张量
  double val = 1.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        tensor(i, j, k) = val++;
      }
    }
  }

  auto tensor_sub = tensor.span(1,             // 第一维 第二个
                                slice{1, -1},  // 第二维 第2-3
                                all()          // 所有第三维
  );

  std::cout << "3D张量子视图[1, 1:-1(2), 0:4]的形状: ";
  std::cout << tensor_sub.extent(0) << "x" << tensor_sub.extent(1) << std::endl;

  // 测试7: 使用span创建mdvector
  std::cout << "\n=== 测试7: 通过span创建子mdvector ===" << std::endl;
  std::cout << "3D张量子视图[1, 1:-1(2), 0:4] 及 +=10 的内容: \n";
  mdvector<double, 2> aa = tensor_sub;
  mdvector<double, 2> bb = tensor_sub + 10.0;
  aa.show_data_matrix_style();
  // 17 18 19 20
  // 21 22 23 24
  std::cout << std::endl;
  bb.show_data_matrix_style();
  // 27 28 29 30
  // 31 32 33 34

  // 测试8: 快速根据节点三维坐标计算10个梁的长度
  std::cout << "\n=== 测试8: 快速根据节点三维坐标计算10个梁的长度 ===" << std::endl;
  mdvector<double, 2> pos_info({3, 11});
  for (int i = 0; i < 11; i++) {
    pos_info(0, i) = i * (10 + i) + 1;
    pos_info(1, i) = i * (10 + i) + 2;
    pos_info(2, i) = i * (10 + i) + 3;
  }
  md::span<double, 1> x1 = pos_info.span(0, slice(0, -2));
  md::span<double, 1> y1 = pos_info.span(1, slice(0, -2));
  md::span<double, 1> z1 = pos_info.span(2, slice(0, -2));
  md::span<double, 1> x2 = pos_info.span(0, slice(1, -1));
  md::span<double, 1> y2 = pos_info.span(1, slice(1, -1));
  md::span<double, 1> z2 = pos_info.span(2, slice(1, -1));
  mdvector<double, 1> length = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
  length = length.sqrt();
  length.show_data_matrix_style();

  return 0;
}