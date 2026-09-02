#pragma once

#include <type_traits>

#include "binary_expr.h"

namespace md {

#define MD_DEFINE_BINARY_EXPRESSION_OPERATOR(symbol, operation)                     \
  template <typename L, typename LT, typename R, typename RT>                       \
  auto operator symbol(const base_expr<L, LT>& lhs, const base_expr<R, RT>& rhs) {  \
    using result_type = std::common_type_t<LT, RT>;                                 \
    return binary_expr<result_type, L, R, operation>(lhs.derived(), rhs.derived()); \
  }                                                                                 \
  template <typename L, typename LT, typename S>                                    \
    requires std::is_arithmetic_v<S>                                                \
  auto operator symbol(const base_expr<L, LT>& lhs, S rhs) {                        \
    using result_type = std::common_type_t<LT, S>;                                  \
    return binary_expr<result_type, L, S, operation>(lhs.derived(), rhs);           \
  }                                                                                 \
  template <typename S, typename R, typename RT>                                    \
    requires std::is_arithmetic_v<S>                                                \
  auto operator symbol(S lhs, const base_expr<R, RT>& rhs) {                        \
    using result_type = std::common_type_t<S, RT>;                                  \
    return binary_expr<result_type, S, R, operation>(lhs, rhs.derived());           \
  }

MD_DEFINE_BINARY_EXPRESSION_OPERATOR(+, Add)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(-, Sub)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(*, Mul)
MD_DEFINE_BINARY_EXPRESSION_OPERATOR(/, Div)
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

template <typename L, typename LT, typename R, typename RT>
auto logical_and(const base_expr<L, LT>& lhs, const base_expr<R, RT>& rhs) {
  using result_type = std::common_type_t<LT, RT>;
  return binary_expr<result_type, L, R, LogicalAnd>(lhs.derived(), rhs.derived());
}

template <typename L, typename LT, typename R, typename RT>
auto logical_or(const base_expr<L, LT>& lhs, const base_expr<R, RT>& rhs) {
  using result_type = std::common_type_t<LT, RT>;
  return binary_expr<result_type, L, R, LogicalOr>(lhs.derived(), rhs.derived());
}

template <typename L, typename LT, typename R, typename RT>
auto logical_xor(const base_expr<L, LT>& lhs, const base_expr<R, RT>& rhs) {
  using result_type = std::common_type_t<LT, RT>;
  return binary_expr<result_type, L, R, LogicalXor>(lhs.derived(), rhs.derived());
}

}  // namespace md
