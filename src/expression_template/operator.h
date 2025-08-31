#ifndef __MDVECTOR_OPERATOR_H__
#define __MDVECTOR_OPERATOR_H__

#include "calculation_expr.h"

namespace md {

// 向量 + 向量
template <class L, class R, class Policy>
auto operator+(const tensor_expr<L, Policy>& lhs, const tensor_expr<R, Policy>& rhs) {
  return calculation_expr<L, R, Policy, Add>(lhs.derived(), rhs.derived());
}

// 向量 + 标量
template <class L, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(const tensor_expr<L, Policy>& lhs, const T& rhs) {
  return calculation_expr<L, T, Policy, Add>(lhs.derived(), rhs);
}

// 标量 + 向量
template <class R, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(T lhs, const tensor_expr<R, Policy>& rhs) {
  return calculation_expr<T, R, Policy, Add>(lhs, rhs.derived());
}

// 向量 - 向量
template <class L, class R, class Policy>
auto operator-(const tensor_expr<L, Policy>& lhs, const tensor_expr<R, Policy>& rhs) {
  return calculation_expr<L, R, Policy, Sub>(lhs.derived(), rhs.derived());
}

// 向量 - 标量
template <class L, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(const tensor_expr<L, Policy>& lhs, T rhs) {
  return calculation_expr<L, T, Policy, Sub>(lhs.derived(), rhs);
}

// 标量 - 向量
template <class R, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(T lhs, const tensor_expr<R, Policy>& rhs) {
  return calculation_expr<T, R, Policy, Sub>(lhs, rhs.derived());
}

// 向量 * 向量
template <class L, class R, class Policy>
auto operator*(const tensor_expr<L, Policy>& lhs, const tensor_expr<R, Policy>& rhs) {
  return calculation_expr<L, R, Policy, Mul>(lhs.derived(), rhs.derived());
}

// 向量 * 标量
template <class L, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(const tensor_expr<L, Policy>& lhs, T rhs) {
  return calculation_expr<L, T, Policy, Mul>(lhs.derived(), rhs);
}

// 标量 * 向量
template <class R, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(T lhs, const tensor_expr<R, Policy>& rhs) {
  return calculation_expr<T, R, Policy, Mul>(lhs, rhs.derived());
}

// 向量 / 向量
template <class L, class R, class Policy>
auto operator/(const tensor_expr<L, Policy>& lhs, const tensor_expr<R, Policy>& rhs) {
  return calculation_expr<L, R, Policy, Div>(lhs.derived(), rhs.derived());
}

// 向量 / 标量
template <class L, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(const tensor_expr<L, Policy>& lhs, T rhs) {
  return calculation_expr<L, T, Policy, Mul>(lhs.derived(), static_cast<T>(1.0) / rhs);
}

// 标量 / 向量
template <class R, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(T lhs, const tensor_expr<R, Policy>& rhs) {
  return calculation_expr<T, R, Policy, Div>(lhs, rhs.derived());
}

}  // namespace md

#endif  // __MDVECTOR_OPERATOR_H__