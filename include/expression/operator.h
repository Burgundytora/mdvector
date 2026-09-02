#pragma once

#include "binary_expr.h"

namespace md {

// 向量 + 向量
template <typename T, typename L, typename R>
auto operator+(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, L, R, Add>(lhs.derived(), rhs.derived());
}

// 向量 + 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(const base_expr<L, T>& lhs, const T& rhs) {
  return binary_expr<T, L, T, Add>(lhs.derived(), rhs);
}

// 标量 + 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator+(T lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, T, R, Add>(lhs, rhs.derived());
}

// 向量 - 向量
template <typename T, typename L, typename R>
auto operator-(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, L, R, Sub>(lhs.derived(), rhs.derived());
}

// 向量 - 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(const base_expr<L, T>& lhs, T rhs) {
  return binary_expr<T, L, T, Sub>(lhs.derived(), rhs);
}

// 标量 - 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator-(T lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, T, R, Sub>(lhs, rhs.derived());
}

// 向量 * 向量
template <typename T, typename L, typename R>
auto operator*(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, L, R, Mul>(lhs.derived(), rhs.derived());
}

// 向量 * 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(const base_expr<L, T>& lhs, T rhs) {
  return binary_expr<T, L, T, Mul>(lhs.derived(), rhs);
}

// 标量 * 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator*(T lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, T, R, Mul>(lhs, rhs.derived());
}

// 向量 / 向量
template <typename T, typename L, typename R>
auto operator/(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, L, R, Div>(lhs.derived(), rhs.derived());
}

// 向量 / 标量
template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(const base_expr<L, T>& lhs, T rhs) {
  return binary_expr<T, L, T, Div>(lhs.derived(), rhs);
}

// 标量 / 向量
template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto operator/(T lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, T, R, Div>(lhs, rhs.derived());
}

#define MD_DEFINE_BINARY_EXPRESSION_OPERATOR(symbol, operation)                           \
  template <typename T, typename L, typename R>                                           \
  auto operator symbol(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {          \
    return binary_expr<T, L, R, operation>(lhs.derived(), rhs.derived());                 \
  }                                                                                       \
  template <typename L, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  auto operator symbol(const base_expr<L, T>& lhs, T rhs) {                               \
    return binary_expr<T, L, T, operation>(lhs.derived(), rhs);                           \
  }                                                                                       \
  template <typename R, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>> \
  auto operator symbol(T lhs, const base_expr<R, T>& rhs) {                               \
    return binary_expr<T, T, R, operation>(lhs, rhs.derived());                           \
  }

MD_DEFINE_BINARY_EXPRESSION_OPERATOR(==, Equal)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(!=, NotEqual)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(<, Less)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(<=, LessEqual)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(>, Greater)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(>=, GreaterEqual)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(&&, LogicalAnd)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(||, LogicalOr)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(&, LogicalAnd)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(|, LogicalOr)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(^, LogicalXor)

#undef MD_DEFINE_BINARY_EXPRESSION_OPERATOR

template <typename Derived, typename T>
auto operator!(const base_expr<Derived, T>& expr) {
  return binary_expr<T, Derived, T, Equal>(expr.derived(), T{});
}

template <typename L, typename R, typename T>
auto logical_and(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, L, R, LogicalAnd>(lhs.derived(), rhs.derived());
}

template <typename L, typename R, typename T>
auto logical_or(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, L, R, LogicalOr>(lhs.derived(), rhs.derived());
}

template <typename L, typename R, typename T>
auto logical_xor(const base_expr<L, T>& lhs, const base_expr<R, T>& rhs) {
  return binary_expr<T, L, R, LogicalXor>(lhs.derived(), rhs.derived());
}

}  // namespace md
