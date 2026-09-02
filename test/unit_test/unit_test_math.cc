
#include "include_md_all.h"

int main(int args, char *argv[]) {
  std::cout << "\nVerification:" << std::endl;

  try {
    md::vector<double, 2> a({2, 3});
    a.set_arange(0.1, 0.1);

    std::cout << "mdvector: print 0.1~0.6 : ";
    a.print();

    auto sin_expr = md::sin(a + 0.1);
    md::vector<double, 2> temp111 = sin_expr;
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
    md::vector<double, 1> span_sin = sin(a.span(0, all()));
    for (const auto &it : span_sin) {
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
    md::vector<double, 1> span_pow = pow(a.span(1, all()), 3);
    for (const auto &it : span_pow) {
      std::cout << it << " ";
    }
    std::cout << "\n";

    std::cout << "span: print abs(-0.1~-0.3) : ";
    md::vector<double, 1> span_abs = abs(-1.0 * a.span(0, all()));
    for (const auto &it : span_abs) {
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

    auto length_expr = hypot(x2 - x1, y2 - y1, z2 - z1);
    length = length_expr;
    length.print();

    // Store compound math expressions and evaluate them in a later statement.
    auto math_expr = md::pow(a + 0.25, 2.0) + md::fmod(a + 1.0, 0.4);
    md::vector<double, 2> math_result = math_expr;
    for (size_t i = 0; i < a.size(); ++i) {
      const double expected = std::pow(a.data()[i] + 0.25, 2.0) + std::fmod(a.data()[i] + 1.0, 0.4);
      if (std::abs(math_result.data()[i] - expected) > 1e-12) return 1;
    }

    auto hypot_expr = md::hypot(a + 1.0, a + 2.0);
    md::vector<double, 2> hypot_result = hypot_expr;
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::abs(hypot_result.data()[i] - std::hypot(a.data()[i] + 1.0, a.data()[i] + 2.0)) > 1e-12) return 1;
    }

    const double expression_sum = md::sum(a + 1.0);
    const double expression_mean = md::mean(md::sqrt(a + 1.0));
    double expected_sum = 0.0;
    double expected_mean = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      expected_sum += a.data()[i] + 1.0;
      expected_mean += std::sqrt(a.data()[i] + 1.0);
    }
    expected_mean /= static_cast<double>(a.size());
    if (std::abs(expression_sum - expected_sum) > 1e-12 || std::abs(expression_mean - expected_mean) > 1e-12) return 1;

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
