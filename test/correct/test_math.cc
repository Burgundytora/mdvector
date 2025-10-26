#include <string>

#if defined(_WIN32)
#include <windows.h>  // 添加Windows头文件
#endif

#include "mdvector.h"

using md::all;
using md::slice;
using md::span;

int main(int args, char *argv[]) {
#if defined(_WIN32)
  // 设置控制台输出为UTF-8编码
  SetConsoleOutputCP(65001);
#endif

  std::cout << "\nVerification:" << std::endl;

  try {
    vector_2d<double> a({2, 3});
    a.fill(0.1);

    double val = 0.1;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        a(i, j) = val;
        val += 0.1;
      }
    }

    std::cout << "mdvector: print 0.1~0.6 : ";
    a.print();

    auto temp111 = md::sin(a + 0.1);
    std::cout << "mdvector: print sin (0.1~0.6 + 0.1) : ";
    temp111.print();

    vector_2d<double> a_cos_plus_10 = md::cos(a) + 10.0;
    std::cout << "mdvector: print cos(0.1~0.6)+10 : ";
    for (const auto &it : a_cos_plus_10) {
      std::cout << it << " ";
    }
    std::cout << "\n";

    vector_2d<double> a_exp_2_plus_1 = exp(a, 2) + 1.0;
    std::cout << "mdvector: print exp 2^(0.1~0.6)+1 : ";
    for (const auto &it : a_exp_2_plus_1) {
      std::cout << it << " ";
    }
    std::cout << "\n";

    std::cout << "span: print sin(0.1~0.3) : ";
    for (const auto &it : sin(a.span(0, all()))) {
      std::cout << it << " ";
    }
    std::cout << "\n";

    std::cout << "span: print sin(0.1~0.3)+10 : ";
    vector_1d<double> temp = sin(a.span(0, all())) + 10.0;
    for (const auto &it : temp) {
      std::cout << it << " ";
    }
    std::cout << "\n";

    std::cout << "span: print (0.4~0.6)^3 : ";
    for (const auto &it : pow(a.span(1, all()), 3)) {
      std::cout << it << " ";
    }
    std::cout << "\n";

    std::cout << "span: print abs(-0.1~-0.3) : ";
    for (const auto &it : abs(-1.0 * a.span(0, all()))) {
      std::cout << it << " ";
    }
    std::cout << "\n";

    using md::max;
    using md::mean;
    using md::median;
    using md::min;
    using md::standard_deviation;
    std::cout << "span: mean [(0.1~0.6)^3] : " << mean(pow(a, 3)) << "\n";
    std::cout << "span: max [(0.4~0.6)^3] : " << max(pow(a.span(1, all()), 3)) << "\n";
    std::cout << "span: min [(0.4~0.6)^3] : " << min(pow(a.span(1, all()), 3)) << "\n";
    std::cout << "span: median [(0.1~0.6)^3] : " << median(pow(a, 3)) << "\n";
    std::cout << "span: std [(0.4~0.6)^3] : " << standard_deviation(pow(a.span(1, all()), 3)) << "\n";
    std::cout << "\n";

    // 测试: 快速根据节点三维坐标计算10个梁的长度
    std::cout << "\n=== 测试8: 快速根据节点三维坐标计算10个梁的长度 ===" << std::endl;
    mdvector<double, 2> pos_info({3, 11});
    for (int i = 0; i < 11; i++) {
      pos_info(0, i) = i * (10 + i) + 1;
      pos_info(1, i) = i * (10 + i) + 2;
      pos_info(2, i) = i * (10 + i) + 3;
    }
    pos_info.print();
    md::span<double, 1> x1 = pos_info.span(0, slice(0, -2));
    md::span<double, 1> y1 = pos_info.span(1, slice(0, -2));
    md::span<double, 1> z1 = pos_info.span(2, slice(0, -2));
    md::span<double, 1> x2 = pos_info.span(0, slice(1, -1));
    md::span<double, 1> y2 = pos_info.span(1, slice(1, -1));
    md::span<double, 1> z2 = pos_info.span(2, slice(1, -1));
    mdvector<double, 1> length;

    // length = sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0) + pow(z2 - z1, 2.0));
    length = hypot(x2 - x1, y2 - y1, z2 - z1);
    length.print();

  } catch (const std::runtime_error &e) {
    std::cout << "捕获异常: " << e.what() << std::endl;
  }

  std::cout << "math test down. \n";

  return 0;
}