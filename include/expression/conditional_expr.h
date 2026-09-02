#pragma once

#include <utility>

#include "extract_layout.h"
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
      : condition_(condition), true_expr_(true_expr), false_expr_(false_expr) {}

  size_t used_size() const noexcept { return condition_.used_size(); }
  size_t size() const noexcept { return condition_.size(); }
  auto extents() const noexcept { return condition_.extents(); }

  template <typename U>
  typename simd<U>::type load_simd(size_t i) const {
    return simd_op_select<U>(condition_.template load_simd<U>(i), true_expr_.template load_simd<U>(i),
                             false_expr_.template load_simd<U>(i));
  }

  template <typename U>
  typename simd<U>::type load_simd_mask(size_t i) const {
    return simd_op_select<U>(condition_.template load_simd_mask<U>(i), true_expr_.template load_simd_mask<U>(i),
                             false_expr_.template load_simd_mask<U>(i));
  }

  T operator[](size_t i) const {
    return scalar_value(condition_, i) != T{} ? scalar_value(true_expr_, i) : scalar_value(false_expr_, i);
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

template <typename Condition, typename TrueExpr, typename FalseExpr, typename T>
inline auto where(const base_expr<Condition, T>& condition, const base_expr<TrueExpr, T>& true_expr,
                  const base_expr<FalseExpr, T>& false_expr) {
  return ternary_expr<T, Condition, TrueExpr, FalseExpr>(condition.derived(), true_expr.derived(),
                                                         false_expr.derived());
}

template <typename Condition, typename TrueExpr, typename T>
inline auto where(const base_expr<Condition, T>& condition, const base_expr<TrueExpr, T>& true_expr, T false_value) {
  return ternary_expr<T, Condition, TrueExpr, T>(condition.derived(), true_expr.derived(), false_value);
}

template <typename Condition, typename FalseExpr, typename T>
inline auto where(const base_expr<Condition, T>& condition, T true_value, const base_expr<FalseExpr, T>& false_expr) {
  return ternary_expr<T, Condition, T, FalseExpr>(condition.derived(), true_value, false_expr.derived());
}

template <typename Condition, typename T>
inline auto where(const base_expr<Condition, T>& condition, T true_value, T false_value) {
  return ternary_expr<T, Condition, T, T>(condition.derived(), true_value, false_value);
}

template <typename... Args>
inline auto select(Args&&... args) {
  return where(std::forward<Args>(args)...);
}

}  // namespace md
