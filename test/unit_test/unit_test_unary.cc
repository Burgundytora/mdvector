#include "include_md_all.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string_view>

template <typename Expr, typename Fn>
bool check_unary(std::string_view name, const md::base_expr<Expr, double>& expr, const md::vector<double, 1>& input,
                 Fn expected_fn) {
  md::vector<double, 1> actual = expr;
  for (size_t i = 0; i < input.size(); ++i) {
    const double expected = expected_fn(input[i]);
    const double tolerance = 1e-12 * std::max(1.0, std::abs(expected));
    if ((std::isnan(expected) && !std::isnan(actual[i])) ||
        (!std::isnan(expected) && std::abs(actual[i] - expected) > tolerance)) {
      std::cerr << name << " failed at " << i << ": expected " << expected << ", got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

int main() {
  constexpr size_t pack_count = 2;
  constexpr size_t size = md::simd<double>::pack_size * pack_count;

  md::vector<double, 1> unit(size);
  md::vector<double, 1> positive(size);
  md::vector<double, 1> negative(size);
  for (size_t i = 0; i < size; ++i) {
    unit[i] = 0.2 + static_cast<double>(i) / static_cast<double>(size * 2);
    positive[i] = 1.2 + static_cast<double>(i) / static_cast<double>(size);
    negative[i] = -1.8 + static_cast<double>(i) / static_cast<double>(size);
  }

  // Keep the binary nodes alive so all checks exercise unary expression nodes,
  // including the SIMD load path rather than the legacy eager math functions.
  auto unit_expr = unit + 0.0;
  auto positive_expr = positive + 0.0;
  auto negative_expr = negative + 0.0;

  bool ok = true;
#define CHECK_UNARY(name, expression, input, expected) \
  ok = check_unary(#name, md::name(expression), input, [](double x) { return expected; }) && ok

  CHECK_UNARY(abs, negative_expr, negative, std::abs(x));
  CHECK_UNARY(sqrt, positive_expr, positive, std::sqrt(x));
  CHECK_UNARY(cbrt, negative_expr, negative, std::cbrt(x));
  CHECK_UNARY(rsqrt, positive_expr, positive, 1.0 / std::sqrt(x));
  CHECK_UNARY(exp, unit_expr, unit, std::exp(x));
  CHECK_UNARY(exp2, unit_expr, unit, std::exp2(x));
  CHECK_UNARY(expm1, unit_expr, unit, std::expm1(x));
  CHECK_UNARY(log, positive_expr, positive, std::log(x));
  CHECK_UNARY(log2, positive_expr, positive, std::log2(x));
  CHECK_UNARY(log10, positive_expr, positive, std::log10(x));
  CHECK_UNARY(log1p, unit_expr, unit, std::log1p(x));
  CHECK_UNARY(sin, unit_expr, unit, std::sin(x));
  CHECK_UNARY(cos, unit_expr, unit, std::cos(x));
  CHECK_UNARY(tan, unit_expr, unit, std::tan(x));
  CHECK_UNARY(asin, unit_expr, unit, std::asin(x));
  CHECK_UNARY(acos, unit_expr, unit, std::acos(x));
  CHECK_UNARY(atan, unit_expr, unit, std::atan(x));
  CHECK_UNARY(sinh, unit_expr, unit, std::sinh(x));
  CHECK_UNARY(cosh, unit_expr, unit, std::cosh(x));
  CHECK_UNARY(tanh, unit_expr, unit, std::tanh(x));
  CHECK_UNARY(asinh, unit_expr, unit, std::asinh(x));
  CHECK_UNARY(acosh, positive_expr, positive, std::acosh(x));
  CHECK_UNARY(atanh, unit_expr, unit, std::atanh(x));
  CHECK_UNARY(floor, negative_expr, negative, std::floor(x));
  CHECK_UNARY(ceil, negative_expr, negative, std::ceil(x));
  CHECK_UNARY(trunc, negative_expr, negative, std::trunc(x));
  CHECK_UNARY(round, negative_expr, negative, std::round(x));
  CHECK_UNARY(erf, unit_expr, unit, std::erf(x));
  CHECK_UNARY(erfc, unit_expr, unit, std::erfc(x));
  CHECK_UNARY(tgamma, positive_expr, positive, std::tgamma(x));
  CHECK_UNARY(lgamma, positive_expr, positive, std::lgamma(x));

#undef CHECK_UNARY

  ok = check_unary("neg", -unit, unit, [](double x) { return -x; }) && ok;

  if (ok) {
    std::cout << "unary test down." << "\n";
  }

  return ok ? 0 : 1;
}
