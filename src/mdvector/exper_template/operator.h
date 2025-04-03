#ifndef __MDVECTOR_OPERATOR_H__
#define __MDVECTOR_OPERATOR_H__

#include "calculation_expr.h"

// ======================== 运算符重载 ========================
// 向量 + 向量
template <typename L, typename R>
AddExpr<L, R> operator+(const TensorExpr<L>& lhs, const TensorExpr<R>& rhs) {
  return AddExpr<L, R>(lhs.derived(), rhs.derived());
}

// 向量 + 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(const TensorExpr<L>& lhs, T rhs) {
  return AddExpr<L, ScalarWrapper<T>>(lhs.derived(), ScalarWrapper<T>(rhs));
}

// 标量 + 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(T lhs, const TensorExpr<R>& rhs) {
  return AddExpr<ScalarWrapper<T>, R>(ScalarWrapper<T>(lhs), rhs.derived());
}

// 向量 - 向量
template <typename L, typename R>
SubExpr<L, R> operator-(const TensorExpr<L>& lhs, const TensorExpr<R>& rhs) {
  return SubExpr<L, R>(lhs.derived(), rhs.derived());
}

// 向量 - 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(const TensorExpr<L>& lhs, T rhs) {
  return SubExpr<L, ScalarWrapper<T>>(lhs.derived(), ScalarWrapper<T>(rhs));
}

// 标量 - 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(T lhs, const TensorExpr<R>& rhs) {
  return SubExpr<ScalarWrapper<T>, R>(ScalarWrapper<T>(lhs), rhs.derived());
}

// 向量 * 向量
template <typename L, typename R>
MulExpr<L, R> operator*(const TensorExpr<L>& lhs, const TensorExpr<R>& rhs) {
  return MulExpr<L, R>(lhs.derived(), rhs.derived());
}

// 向量 * 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(const TensorExpr<L>& lhs, T rhs) {
  return MulExpr<L, ScalarWrapper<T>>(lhs.derived(), ScalarWrapper<T>(rhs));
}

// 标量 * 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(T lhs, const TensorExpr<R>& rhs) {
  return MulExpr<ScalarWrapper<T>, R>(ScalarWrapper<T>(lhs), rhs.derived());
}

// 向量 / 向量
template <typename L, typename R>
DivExpr<L, R> operator/(const TensorExpr<L>& lhs, const TensorExpr<R>& rhs) {
  return DivExpr<L, R>(lhs.derived(), rhs.derived());
}

// 向量 / 标量
// 优化为标量倒数乘法 win x64有bug？
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(const TensorExpr<L>& lhs, T rhs) {
  return DivExpr<L, ScalarWrapper<T>>(lhs.derived(), ScalarWrapper<T>(rhs));
}

// 标量 / 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(T lhs, const TensorExpr<R>& rhs) {
  return DivExpr<ScalarWrapper<T>, R>(ScalarWrapper<T>(lhs), rhs.derived());
}

#endif  // __OPERATOR_H__