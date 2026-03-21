#ifndef __MDVECTOR_MATH_FUNCTION__
#define __MDVECTOR_MATH_FUNCTION__

#include <cmath>

#include "common/mdvector_def.h"

// 独立数学函数模板
namespace md {

// 数学函数返回一个新的mdvector做为临时值
#define DEFINE_BASE_MD_MATH_FUNC(name, op)                                                                 \
  template <MathContainer Container>                                                                       \
  auto name(const Container& c) noexcept {                                                                 \
    md::vector<typename Container::value_type, Container::rank_, typename Container::layout_type> res = c; \
    std::transform(res.begin(), res.end(), res.begin(),                                                    \
                   [](Container::value_type val) noexcept { return std::op(val); });                       \
    return res;                                                                                            \
  }

DEFINE_BASE_MD_MATH_FUNC(cos, cos);
DEFINE_BASE_MD_MATH_FUNC(cosh, cosh);
DEFINE_BASE_MD_MATH_FUNC(acos, acos);
DEFINE_BASE_MD_MATH_FUNC(sin, sin);
DEFINE_BASE_MD_MATH_FUNC(sinh, sinh);
DEFINE_BASE_MD_MATH_FUNC(asin, asin);
DEFINE_BASE_MD_MATH_FUNC(tan, tan);
DEFINE_BASE_MD_MATH_FUNC(atan, atan);
DEFINE_BASE_MD_MATH_FUNC(tanh, tanh);
DEFINE_BASE_MD_MATH_FUNC(abs, abs);
DEFINE_BASE_MD_MATH_FUNC(sqrt, sqrt);
DEFINE_BASE_MD_MATH_FUNC(cbrt, cbrt);
DEFINE_BASE_MD_MATH_FUNC(log10, log10);
DEFINE_BASE_MD_MATH_FUNC(ln, log);
DEFINE_BASE_MD_MATH_FUNC(ceil, ceil);
DEFINE_BASE_MD_MATH_FUNC(floor, floor);
DEFINE_BASE_MD_MATH_FUNC(trunc, trunc);

#undef DEFINE_BASE_MD_MATH_FUNC

template <MathContainer Container, typename T>
auto exp(const Container& c, T y) {
  md::vector<typename Container::value_type, Container::rank_, typename Container::layout_type> res = c;
  std::transform(res.begin(), res.end(), res.begin(), [y](double val) noexcept { return std::pow(y, val); });
  return res;
}

template <MathContainer Container, typename T>
auto pow(const Container& c, T y) {
  md::vector<typename Container::value_type, Container::rank_, typename Container::layout_type> res = c;
  std::transform(res.begin(), res.end(), res.begin(), [y](double val) noexcept { return std::pow(val, y); });
  return res;
}

template <MathContainer Container, typename T>
auto fmod(const Container& c, T y) {
  md::vector<typename Container::value_type, Container::rank_, typename Container::layout_type> res = c;
  std::transform(res.begin(), res.end(), res.begin(), [y](double val) noexcept { return std::fmod(val, y); });
  return res;
}

template <MathContainer Container1, MathContainer Container2>
auto hypot(const Container1& x, const Container2& y) {
  using value_type = typename Container1::value_type;
  md::vector<value_type, Container1::rank_, typename Container1::layout_type> res = x;
  md::vector<value_type, Container1::rank_, typename Container1::layout_type> y_vec = y;

  auto x_it = res.begin();
  auto y_it = y_vec.begin();
  auto res_it = res.begin();

  for (; x_it != res.end() && y_it != y_vec.end(); ++x_it, ++y_it, ++res_it) {
    *res_it = std::hypot(*x_it, *y_it);
  }

  return res;
}

// 三个容器的 hypot
template <MathContainer Container1, MathContainer Container2, MathContainer Container3>
auto hypot(const Container1& x, const Container2& y, const Container3& z) {
  using value_type = typename Container1::value_type;
  md::vector<value_type, Container1::rank_, typename Container1::layout_type> res = x;
  md::vector<value_type, Container1::rank_, typename Container1::layout_type> y_vec = y;
  md::vector<value_type, Container1::rank_, typename Container1::layout_type> z_vec = z;

  auto x_it = res.begin();
  auto y_it = y_vec.begin();
  auto z_it = z_vec.begin();
  auto res_it = res.begin();

  for (; x_it != res.end() && y_it != y_vec.end() && z_it != z_vec.end(); ++x_it, ++y_it, ++z_it, ++res_it) {
    *res_it = std::hypot(*x_it, *y_it, *z_it);
  }

  return res;
}

template <StatisticContainer Container>
auto sum(const Container& c) {
  return std::reduce(c.begin(), c.end());
}

template <StatisticContainer Container>
auto prod(const Container& c) {
  return std::reduce(c.begin(), c.end(), typename Container::value_type(1), std::multiplies<>());
}

template <StatisticContainer Container>
auto max(const Container& c) {
  return *std::max_element(c.begin(), c.end());
}

template <StatisticContainer Container>
auto min(const Container& c) {
  return *std::min_element(c.begin(), c.end());
}

template <StatisticContainer Container>
auto mean(const Container& c) {
  return sum(c) / static_cast<typename Container::value_type>(c.size());
}

template <StatisticContainer Container>
auto variance(const Container& c) {
  auto m = mean(c);
  double sum_sq = std::accumulate(c.begin(), c.end(), 0.0, [m](double acc, auto val) {
    double diff = static_cast<double>(val) - static_cast<double>(m);
    return acc + diff * diff;
  });
  return static_cast<typename Container::value_type>(sum_sq / (c.size() - 1));
}

template <StatisticContainer Container>
auto standard_deviation(const Container& c) {
  return static_cast<typename Container::value_type>(std::sqrt(variance(c)));
}

template <StatisticContainer Container>
auto median(const Container& c) {  // 按值传递以进行排序
  std::vector<typename Container::value_type> vec(c.size());
  vec.assign(std::begin(c), std::end(c));
  std::sort(vec.begin(), vec.end());
  size_t n = vec.size();
  if (n % 2 == 0) {
    return (vec[n / 2 - 1] + vec[n / 2]) / typename Container::value_type(2);
  } else {
    return vec[n / 2];
  }
}

// 最大值索引
template <StatisticContainer Container>
size_t max_index(const Container& c) {
  return std::distance(c.begin(), std::max_element(c.begin(), c.end()));
}

// 最小值索引
template <StatisticContainer Container>
size_t min_index(const Container& c) {
  return std::distance(c.begin(), std::min_element(c.begin(), c.end()));
}

// 绝对值最大值
template <StatisticContainer Container>
auto abs_max(const Container& c) {
  if (c.empty()) return typename Container::value_type(0);
  return *std::max_element(c.begin(), c.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); });
}

// 绝对值最小值
template <StatisticContainer Container>
auto abs_min(const Container& c) {
  if (c.empty()) return typename Container::value_type(0);
  return *std::min_element(c.begin(), c.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); });
}

}  // namespace md

#endif  // __MDVECTOR_MATH_FUNCTION__