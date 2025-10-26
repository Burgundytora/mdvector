#ifndef __MDVECTOR_MATH_FUNCTION_H__
#define __MDVECTOR_MATH_FUNCTION_H__

#include <cmath>

#include "mdspan_little.h"

// 前向声明
template <typename T, size_t Rank, typename Layout = std::layout_right>
class mdvector;

#include <concepts>
#include <type_traits>

// 定义容器概念
template <typename T>
concept MathContainer = requires(T c) {
  { T::rank_ } -> std::convertible_to<size_t>;
};

// 独立数学函数模板
namespace md {

// 数学函数返回一个新的mdvector做为临时值
#define DEFINE_BASE_MD_MATH_FUNC(name, op)                                                               \
  template <MathContainer Container>                                                                     \
  auto name(const Container& c) noexcept {                                                               \
    mdvector<typename Container::value_type, Container::rank_, typename Container::layout_type> res = c; \
    std::transform(res.begin(), res.end(), res.begin(),                                                  \
                   [](Container::value_type val) noexcept { return std::op(val); });                     \
    return res;                                                                                          \
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
  mdvector<typename Container::value_type, Container::rank_, typename Container::layout_type> res = c;
  std::transform(res.begin(), res.end(), res.begin(), [y](double val) noexcept { return std::pow(y, val); });
  return res;
}

template <MathContainer Container, typename T>
auto pow(const Container& c, T y) {
  mdvector<typename Container::value_type, Container::rank_, typename Container::layout_type> res = c;
  std::transform(res.begin(), res.end(), res.begin(), [y](double val) noexcept { return std::pow(val, y); });
  return res;
}

template <MathContainer Container, typename T>
auto fmod(const Container& c, T y) {
  mdvector<typename Container::value_type, Container::rank_, typename Container::layout_type> res = c;
  std::transform(res.begin(), res.end(), res.begin(), [y](double val) noexcept { return std::fmod(val, y); });
  return res;
}

template <MathContainer Container1, MathContainer Container2>
auto hypot(const Container1& x, const Container2& y) {
  using value_type = typename Container1::value_type;
  mdvector<value_type, Container1::rank_, typename Container1::layout_type> res = x;
  mdvector<value_type, Container1::rank_, typename Container1::layout_type> y_vec = y;

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
  mdvector<value_type, Container1::rank_, typename Container1::layout_type> res = x;
  mdvector<value_type, Container1::rank_, typename Container1::layout_type> y_vec = y;
  mdvector<value_type, Container1::rank_, typename Container1::layout_type> z_vec = z;

  auto x_it = res.begin();
  auto y_it = y_vec.begin();
  auto z_it = z_vec.begin();
  auto res_it = res.begin();

  for (; x_it != res.end() && y_it != y_vec.end() && z_it != z_vec.end(); ++x_it, ++y_it, ++z_it, ++res_it) {
    *res_it = std::hypot(*x_it, *y_it, *z_it);
  }

  return res;
}

}  // namespace md

#endif  // __MDVECTOR_MATH_FUNCTION_H__