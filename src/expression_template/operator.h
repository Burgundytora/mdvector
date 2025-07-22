#ifndef __MDVECTOR_OPERATOR_H__
#define __MDVECTOR_OPERATOR_H__

#include "calculation_expr.h"

namespace md {

// 向量 + 向量
template <class L, class R, class Policy>
AddExpr<L, R, Policy> operator+(const TensorExpr<L, Policy>& lhs, const TensorExpr<R, Policy>& rhs) {
  return AddExpr<L, R, Policy>(lhs.derived(), rhs.derived());
}

// 向量 + 标量
template <class L, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(const TensorExpr<L, Policy>& lhs, const T& rhs) {
  return AddExpr<L, T, Policy>(lhs.derived(), rhs);
}

// 标量 + 向量
template <class R, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(T lhs, const TensorExpr<R, Policy>& rhs) {
  return AddExpr<T, R, Policy>(lhs, rhs.derived());
}

// 向量 - 向量
template <class L, class R, class Policy>
SubExpr<L, R, Policy> operator-(const TensorExpr<L, Policy>& lhs, const TensorExpr<R, Policy>& rhs) {
  return SubExpr<L, R, Policy>(lhs.derived(), rhs.derived());
}

// 向量 - 标量
template <class L, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(const TensorExpr<L, Policy>& lhs, T rhs) {
  return SubExpr<L, T, Policy>(lhs.derived(), rhs);
}

// 标量 - 向量
template <class R, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(T lhs, const TensorExpr<R, Policy>& rhs) {
  return SubExpr<T, R, Policy>(lhs, rhs.derived());
}

// 向量 * 向量
template <class L, class R, class Policy>
MulExpr<L, R, Policy> operator*(const TensorExpr<L, Policy>& lhs, const TensorExpr<R, Policy>& rhs) {
  return MulExpr<L, R, Policy>(lhs.derived(), rhs.derived());
}

// 向量 * 标量
template <class L, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(const TensorExpr<L, Policy>& lhs, T rhs) {
  return MulExpr<L, T, Policy>(lhs.derived(), rhs);
}

// 标量 * 向量
template <class R, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(T lhs, const TensorExpr<R, Policy>& rhs) {
  return MulExpr<T, R, Policy>(lhs, rhs.derived());
}

// 向量 / 向量
template <class L, class R, class Policy>
DivExpr<L, R, Policy> operator/(const TensorExpr<L, Policy>& lhs, const TensorExpr<R, Policy>& rhs) {
  return DivExpr<L, R, Policy>(lhs.derived(), rhs.derived());
}

// 向量 / 标量
template <class L, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(const TensorExpr<L, Policy>& lhs, T rhs) {
  return MulExpr<L, T, Policy>(lhs.derived(), static_cast<T>(1) / rhs);
}

// 标量 / 向量
template <class R, class T, class Policy, class = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(T lhs, const TensorExpr<R, Policy>& rhs) {
  return DivExpr<T, R, Policy>(lhs, rhs.derived());
}

}  // namespace md

#endif  // __MDVECTOR_OPERATOR_H__