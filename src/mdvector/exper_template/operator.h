#ifndef __MDVECTOR_OPERATOR_H__
#define __MDVECTOR_OPERATOR_H__

#include "calculation_expr.h"

// ======================== 运算符重载 ========================
// 向量 + 向量
template <typename L, typename R, class Policy>
AddExpr<L, R, Policy> operator+(const TensorExpr<L, Policy>& lhs, const TensorExpr<R, Policy>& rhs) {
  return AddExpr<L, R, Policy>(lhs.derived(), rhs.derived());
}

// 向量 + 标量
template <typename L, typename T, class Policy, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(const TensorExpr<L, Policy>& lhs, T rhs) {
  return AddExpr<L, ScalarWrapper<T, Policy>, Policy>(lhs.derived(), ScalarWrapper<T, Policy>(rhs));
}

// 标量 + 向量
template <typename R, typename T, class Policy, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(T lhs, const TensorExpr<R, Policy>& rhs) {
  return AddExpr<ScalarWrapper<T, Policy>, R, Policy>(ScalarWrapper<T, Policy>(lhs), rhs.derived());
}

// 向量 - 向量
template <typename L, typename R, class Policy>
SubExpr<L, R, Policy> operator-(const TensorExpr<L, Policy>& lhs, const TensorExpr<R, Policy>& rhs) {
  return SubExpr<L, R, Policy>(lhs.derived(), rhs.derived());
}

// 向量 - 标量
template <typename L, typename T, class Policy, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(const TensorExpr<L, Policy>& lhs, T rhs) {
  return SubExpr<L, ScalarWrapper<T, Policy>, Policy>(lhs.derived(), ScalarWrapper<T, Policy>(rhs));
}

// 标量 - 向量
template <typename R, typename T, class Policy, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(T lhs, const TensorExpr<R, Policy>& rhs) {
  return SubExpr<ScalarWrapper<T, Policy>, R, Policy>(ScalarWrapper<T, Policy>(lhs), rhs.derived());
}

// 向量 * 向量
template <typename L, typename R, class Policy>
MulExpr<L, R, Policy> operator*(const TensorExpr<L, Policy>& lhs, const TensorExpr<R, Policy>& rhs) {
  return MulExpr<L, R, Policy>(lhs.derived(), rhs.derived());
}

// 向量 * 标量
template <typename L, typename T, class Policy, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(const TensorExpr<L, Policy>& lhs, T rhs) {
  return MulExpr<L, ScalarWrapper<T, Policy>, Policy>(lhs.derived(), ScalarWrapper<T, Policy>(rhs));
}

// 标量 * 向量
template <typename R, typename T, class Policy, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(T lhs, const TensorExpr<R, Policy>& rhs) {
  return MulExpr<ScalarWrapper<T, Policy>, R, Policy>(ScalarWrapper<T, Policy>(lhs), rhs.derived());
}

// 向量 / 向量
template <typename L, typename R, class Policy>
DivExpr<L, R, Policy> operator/(const TensorExpr<L, Policy>& lhs, const TensorExpr<R, Policy>& rhs) {
  return DivExpr<L, R, Policy>(lhs.derived(), rhs.derived());
}

// 向量 / 标量
// 优化为标量倒数乘法 win x64有bug？
template <typename L, typename T, class Policy, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(const TensorExpr<L, Policy>& lhs, T rhs) {
  return DivExpr<L, ScalarWrapper<T, Policy>, Policy>(lhs.derived(), ScalarWrapper<T, Policy>(rhs));
}

// 标量 / 向量
template <typename R, typename T, class Policy, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(T lhs, const TensorExpr<R, Policy>& rhs) {
  return DivExpr<ScalarWrapper<T, Policy>, R, Policy>(ScalarWrapper<T, Policy>(lhs), rhs.derived());
}

#endif  // __OPERATOR_H__