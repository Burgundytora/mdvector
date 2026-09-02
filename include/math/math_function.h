#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "../expression/binary_expr.h"
#include "../expression/unary_expr.h"

namespace md {

// Element-wise mathematical functions. Unary functions are declared in
// unary_expr.h; these overloads cover the binary functions without creating
// an intermediate md::vector.
template <typename Derived, typename T, typename Y>
  requires std::convertible_to<Y, T>
inline auto exp(const base_expr<Derived, T>& exponent, Y base) {
  return binary_expr<T, T, Derived, Pow>(static_cast<T>(base), exponent.derived());
}

template <typename Derived, typename T, typename Y>
  requires std::convertible_to<Y, T>
inline auto pow(const base_expr<Derived, T>& value, Y exponent) {
  return binary_expr<T, Derived, T, Pow>(value.derived(), static_cast<T>(exponent));
}

template <typename L, typename R, typename T>
inline auto pow(const base_expr<L, T>& value, const base_expr<R, T>& exponent) {
  return binary_expr<T, L, R, Pow>(value.derived(), exponent.derived());
}

template <typename Derived, typename T, typename Y>
  requires std::convertible_to<Y, T>
inline auto fmod(const base_expr<Derived, T>& value, Y divisor) {
  return binary_expr<T, Derived, T, Fmod>(value.derived(), static_cast<T>(divisor));
}

template <typename L, typename R, typename T>
inline auto fmod(const base_expr<L, T>& value, const base_expr<R, T>& divisor) {
  return binary_expr<T, L, R, Fmod>(value.derived(), divisor.derived());
}

template <typename L, typename R, typename T>
inline auto hypot(const base_expr<L, T>& x, const base_expr<R, T>& y) {
  return binary_expr<T, L, R, Hypot>(x.derived(), y.derived());
}

template <typename X, typename Y, typename Z, typename T>
inline auto hypot(const base_expr<X, T>& x, const base_expr<Y, T>& y, const base_expr<Z, T>& z) {
  return hypot(hypot(x, y), z);
}

// Generic container reductions are retained for standard iterator-based
// containers. Expression-specific overloads below consume lazy trees directly.
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
  long double sum_sq = std::accumulate(c.begin(), c.end(), 0.0L, [m](long double acc, auto val) {
    const long double diff = static_cast<long double>(val) - static_cast<long double>(m);
    return acc + diff * diff;
  });
  return static_cast<typename Container::value_type>(sum_sq / (c.size() - 1));
}

template <StatisticContainer Container>
auto standard_deviation(const Container& c) {
  return static_cast<typename Container::value_type>(std::sqrt(variance(c)));
}

template <StatisticContainer Container>
auto median(const Container& c) {
  std::vector<typename Container::value_type> values(c.begin(), c.end());
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n % 2 == 0) {
    return (values[n / 2 - 1] + values[n / 2]) / typename Container::value_type(2);
  }
  return values[n / 2];
}

template <StatisticContainer Container>
size_t max_index(const Container& c) {
  return static_cast<size_t>(std::distance(c.begin(), std::max_element(c.begin(), c.end())));
}

template <StatisticContainer Container>
size_t min_index(const Container& c) {
  return static_cast<size_t>(std::distance(c.begin(), std::min_element(c.begin(), c.end())));
}

template <StatisticContainer Container>
auto abs_max(const Container& c) {
  if (c.empty()) return typename Container::value_type(0);
  return *std::max_element(c.begin(), c.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); });
}

template <StatisticContainer Container>
auto abs_min(const Container& c) {
  if (c.empty()) return typename Container::value_type(0);
  return *std::min_element(c.begin(), c.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); });
}

template <typename Derived, typename T>
T sum(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  T result{};
  for (size_t i = 0; i < expr.size(); ++i) result += expr[i];
  return result;
}

template <typename Derived, typename T>
T prod(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  T result{1};
  for (size_t i = 0; i < expr.size(); ++i) result *= expr[i];
  return result;
}

template <typename Derived, typename T>
T max(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  T result = expr[0];
  for (size_t i = 1; i < expr.size(); ++i) result = std::max(result, static_cast<T>(expr[i]));
  return result;
}

template <typename Derived, typename T>
T min(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  T result = expr[0];
  for (size_t i = 1; i < expr.size(); ++i) result = std::min(result, static_cast<T>(expr[i]));
  return result;
}

template <typename Derived, typename T>
T mean(const base_expr<Derived, T>& expression) {
  return sum(expression) / static_cast<T>(expression.size());
}

template <typename Derived, typename T>
T variance(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  const T m = mean(expression);
  long double sum_sq = 0.0L;
  for (size_t i = 0; i < expr.size(); ++i) {
    const long double diff = static_cast<long double>(expr[i]) - static_cast<long double>(m);
    sum_sq += diff * diff;
  }
  return static_cast<T>(sum_sq / (expr.size() - 1));
}

template <typename Derived, typename T>
T standard_deviation(const base_expr<Derived, T>& expression) {
  return static_cast<T>(std::sqrt(variance(expression)));
}

template <typename Derived, typename T>
T median(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  std::vector<T> values(expr.size());
  for (size_t i = 0; i < expr.size(); ++i) values[i] = expr[i];
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n % 2 == 0) return (values[n / 2 - 1] + values[n / 2]) / T(2);
  return values[n / 2];
}

template <typename Derived, typename T>
size_t max_index(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  size_t result = 0;
  for (size_t i = 1; i < expr.size(); ++i) {
    if (expr[result] < expr[i]) result = i;
  }
  return result;
}

template <typename Derived, typename T>
size_t min_index(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  size_t result = 0;
  for (size_t i = 1; i < expr.size(); ++i) {
    if (expr[i] < expr[result]) result = i;
  }
  return result;
}

template <typename Derived, typename T>
T abs_max(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  if (expr.size() == 0) return T{};
  size_t result = 0;
  for (size_t i = 1; i < expr.size(); ++i) {
    if (std::abs(expr[result]) < std::abs(expr[i])) result = i;
  }
  return expr[result];
}

template <typename Derived, typename T>
T abs_min(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  if (expr.size() == 0) return T{};
  size_t result = 0;
  for (size_t i = 1; i < expr.size(); ++i) {
    if (std::abs(expr[i]) < std::abs(expr[result])) result = i;
  }
  return expr[result];
}

}  // namespace md
