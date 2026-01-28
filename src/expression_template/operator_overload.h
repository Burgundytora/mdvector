#ifndef __MDVECTOR_OPERATOR_H__
#define __MDVECTOR_OPERATOR_H__

#include "binary_operation_expr.h"

namespace md {

// 向量 + 向量
template <typename T, typename L, typename R>
auto operator+(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_operation_expr<T, L, R, Add>(lhs.derived(), rhs.derived());
}

// 向量 + 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(const base_expr<L, T>& lhs, const T& rhs) {
  return binary_operation_expr<T, L, T, Add>(lhs.derived(), rhs);
}

// 标量 + 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(T lhs, const base_expr<R, T>& rhs) {
  return binary_operation_expr<T, T, R, Add>(lhs, rhs.derived());
}

// 向量 - 向量
template <typename T, typename L, typename R>
auto operator-(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_operation_expr<T, L, R, Sub>(lhs.derived(), rhs.derived());
}

// 向量 - 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(const base_expr<L, T>& lhs, T rhs) {
  return binary_operation_expr<T, L, T, Sub>(lhs.derived(), rhs);
}

// 标量 - 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(T lhs, const base_expr<R, T>& rhs) {
  return binary_operation_expr<T, T, R, Sub>(lhs, rhs.derived());
}

// 向量 * 向量
template <typename T, typename L, typename R>
auto operator*(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_operation_expr<T, L, R, Mul>(lhs.derived(), rhs.derived());
}

// 向量 * 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(const base_expr<L, T>& lhs, T rhs) {
  return binary_operation_expr<T, L, T, Mul>(lhs.derived(), rhs);
}

// 标量 * 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(T lhs, const base_expr<R, T>& rhs) {
  return binary_operation_expr<T, T, R, Mul>(lhs, rhs.derived());
}

// 向量 / 向量
template <typename T, typename L, typename R>
auto operator/(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_operation_expr<T, L, R, Div>(lhs.derived(), rhs.derived());
}

// 向量 / 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(const base_expr<L, T>& lhs, T rhs) {
  return binary_operation_expr<T, L, T, Mul>(lhs.derived(), static_cast<T>(1.0) / rhs);
}

// 标量 / 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(T lhs, const base_expr<R, T>& rhs) {
  return binary_operation_expr<T, T, R, Div>(lhs, rhs.derived());
}

}  // namespace md

#endif  // __MDVECTOR_OPERATOR_H__