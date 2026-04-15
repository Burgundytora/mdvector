
#include "include_md_all.h"

int main(int args, char *argv[]) {
  std::cout << "\nVerification:" << std::endl;

  try {
    md::vector<double, 2> a({2, 3});
    a.set_arange(0.1, 0.1);

    std::cout << "mdvector: print 0.1~0.6 : ";
    a.print();

    auto temp111 = md::sin(a + 0.1);
    std::cout << "mdvector: print sin (0.1~0.6 + 0.1) : ";
    temp111.print();

    md::vector<double, 2> a_cos_plus_10 = md::cos(a) + 10.0;
    std::cout << "mdvector: print cos(0.1~0.6)+10 : ";
    for (const auto &it : a_cos_plus_10) {
      std::cout << it << " ";
    }
    std::cout << "\n";

    md::vector<double, 2> a_exp_2_plus_1 = exp(a, 2) + 1.0;
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
    md::vector<double, 1> temp = sin(a.span(0, all())) + 10.0;
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

    std::cout << "span: mean [(0.1~0.6)^3] : " << mean(pow(a, 3)) << "\n";
    std::cout << "span: max [(0.4~0.6)^3] : " << max(pow(a.span(1, all()), 3)) << "\n";
    std::cout << "span: min [(0.4~0.6)^3] : " << min(pow(a.span(1, all()), 3)) << "\n";
    std::cout << "span: median [(0.1~0.6)^3] : " << median(pow(a, 3)) << "\n";
    std::cout << "span: std [(0.4~0.6)^3] : " << standard_deviation(pow(a.span(1, all()), 3)) << "\n";
    std::cout << "\n";

    // 测试: 快速根据节点三维坐标计算10个梁的长度
    std::cout << "\n=== 测试8: 快速根据节点三维坐标计算10个梁的长度 ===" << std::endl;
    md::vector<double, 2> pos_info({3, 11});
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
    md::vector<double, 1> length;

    length = hypot(x2 - x1, y2 - y1, z2 - z1);
    length.print();

    // operator-
    std::cout << "\n=== -x2.print ===" << std::endl;
    md::vector<double, 1> op_sub = -x2;
    op_sub.print();

  } catch (const std::runtime_error &e) {
    std::cout << "捕获异常: " << e.what() << std::endl;
  }

  std::cout << "math test down. \n";

  return 0;
}