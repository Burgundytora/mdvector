#pragma once

#include <utility>

#include "extract_layout.h"
#include "expression_shape.h"
#include "../simd/simd_op_select.h"

namespace md {

template <typename T, typename Condition, typename TrueExpr, typename FalseExpr>
class ternary_expr : public base_expr<ternary_expr<T, Condition, TrueExpr, FalseExpr>, T> {
  AutoType<Condition> condition_;
  AutoType<TrueExpr> true_expr_;
  AutoType<FalseExpr> false_expr_;

 public:
  using value_type = T;
  using layout_type = typename Condition::layout_type;
  static constexpr size_t rank_ = Condition::rank_;

  ternary_expr(const Condition& condition, const TrueExpr& true_expr, const FalseExpr& false_expr)
      : condition_(condition), true_expr_(true_expr), false_expr_(false_expr) {
    if constexpr (!std::is_arithmetic_v<TrueExpr>) detail::debug_check_same_shape(condition, true_expr);
    if constexpr (!std::is_arithmetic_v<FalseExpr>) detail::debug_check_same_shape(condition, false_expr);
  }

  size_t used_size() const noexcept { return get_aligned_size<T>(size()); }
  size_t size() const noexcept { return condition_.size(); }
  auto extents() const noexcept { return condition_.extents(); }

  template <typename U>
  typename simd<U>::type load_simd(size_t i) const {
    return simd_op_select<U>(detail::load_operand_simd<U>(condition_, i), detail::load_operand_simd<U>(true_expr_, i),
                             detail::load_operand_simd<U>(false_expr_, i));
  }

  template <typename U>
  typename simd<U>::type load_simd_mask(size_t i) const {
    return load_simd<U>(i);
  }

  T operator[](size_t i) const {
    return scalar_value(condition_, i) != T{} ? scalar_value(true_expr_, i) : scalar_value(false_expr_, i);
  }

  template <typename Dest>
  bool requires_temporary(const Dest& dest) const {
    return condition_.requires_temporary(dest) || true_expr_.requires_temporary(dest) ||
           false_expr_.requires_temporary(dest);
  }

 private:
  template <typename Expr>
  static T scalar_value(const Expr& expr, size_t i) {
    if constexpr (detail::is_expression_node_v<Expr>) {
      return static_cast<T>(expr[i]);
    } else {
      return static_cast<T>(expr.scalar_at(i));
    }
  }
};

template <typename Condition, typename CT, typename TrueExpr, typename TT, typename FalseExpr, typename FT>
inline auto where(const base_expr<Condition, CT>& condition, const base_expr<TrueExpr, TT>& true_expr,
                  const base_expr<FalseExpr, FT>& false_expr) {
  using result_type = std::common_type_t<TT, FT>;
  return ternary_expr<result_type, Condition, TrueExpr, FalseExpr>(condition.derived(), true_expr.derived(),
                                                                   false_expr.derived());
}

template <typename Condition, typename CT, typename TrueExpr, typename TT, typename S>
  requires std::is_arithmetic_v<S>
inline auto where(const base_expr<Condition, CT>& condition, const base_expr<TrueExpr, TT>& true_expr, S false_value) {
  using result_type = std::common_type_t<TT, S>;
  return ternary_expr<result_type, Condition, TrueExpr, S>(condition.derived(), true_expr.derived(), false_value);
}

template <typename Condition, typename CT, typename S, typename FalseExpr, typename FT>
  requires std::is_arithmetic_v<S>
inline auto where(const base_expr<Condition, CT>& condition, S true_value, const base_expr<FalseExpr, FT>& false_expr) {
  using result_type = std::common_type_t<S, FT>;
  return ternary_expr<result_type, Condition, S, FalseExpr>(condition.derived(), true_value, false_expr.derived());
}

template <typename Condition, typename CT, typename TS, typename FS>
  requires(std::is_arithmetic_v<TS> && std::is_arithmetic_v<FS>)
inline auto where(const base_expr<Condition, CT>& condition, TS true_value, FS false_value) {
  using result_type = std::common_type_t<TS, FS>;
  return ternary_expr<result_type, Condition, TS, FS>(condition.derived(), true_value, false_value);
}

template <typename... Args>
inline auto select(Args&&... args) {
  return where(std::forward<Args>(args)...);
}

}  // namespace md
