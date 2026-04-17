// expression/unary_expr.h
#ifndef __MDARRAY_UNARY_EXPR__
#define __MDARRAY_UNARY_EXPR__

#include "base_expr.h"
#include "../simd/simd_op_unary.h"

namespace md {

template <typename Op, typename SubExpr, typename T>
class unary_expr : public base_expr<unary_expr<Op, SubExpr, T>, T> {
  const SubExpr& expr_;

 public:
  using value_type = T;

  explicit unary_expr(const SubExpr& expr) noexcept : expr_(expr) {}

  size_t used_size() const noexcept { return expr_.used_size(); }
  auto extents() const noexcept { return expr_.extents(); }

  template <typename T2>
  auto load_simd(size_t i) const noexcept {
    auto val = expr_.template load_simd<T2>(i);
    return simd_op_unary<T2, Op>(val);
  }

  // 标量访问（用于边界或不支持 SIMD 的情况）
  T operator[](size_t i) const { return scalar_op_unary<T, Op>(expr_[i]); }
};

// ============================================================================
// 运算符重载
// ============================================================================
template <typename Derived, typename T>
inline auto operator-(const base_expr<Derived, T>& expr) {
  return unary_expr<Neg, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto abs(const base_expr<Derived, T>& expr) {
  return unary_expr<Abs, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto sqrt(const base_expr<Derived, T>& expr) {
  return unary_expr<Sqrt, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto cbrt(const base_expr<Derived, T>& expr) {
  return unary_expr<Cbrt, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto rsqrt(const base_expr<Derived, T>& expr) {
  return unary_expr<RSqrt, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto exp(const base_expr<Derived, T>& expr) {
  return unary_expr<Exp, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto exp2(const base_expr<Derived, T>& expr) {
  return unary_expr<Exp2, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto expm1(const base_expr<Derived, T>& expr) {
  return unary_expr<Expm1, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto log(const base_expr<Derived, T>& expr) {
  return unary_expr<Log, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto log2(const base_expr<Derived, T>& expr) {
  return unary_expr<Log2, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto log10(const base_expr<Derived, T>& expr) {
  return unary_expr<Log10, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto log1p(const base_expr<Derived, T>& expr) {
  return unary_expr<Log1p, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto sin(const base_expr<Derived, T>& expr) {
  return unary_expr<Sin, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto cos(const base_expr<Derived, T>& expr) {
  return unary_expr<Cos, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto tan(const base_expr<Derived, T>& expr) {
  return unary_expr<Tan, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto asin(const base_expr<Derived, T>& expr) {
  return unary_expr<Asin, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto acos(const base_expr<Derived, T>& expr) {
  return unary_expr<Acos, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto atan(const base_expr<Derived, T>& expr) {
  return unary_expr<Atan, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto sinh(const base_expr<Derived, T>& expr) {
  return unary_expr<Sinh, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto cosh(const base_expr<Derived, T>& expr) {
  return unary_expr<Cosh, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto tanh(const base_expr<Derived, T>& expr) {
  return unary_expr<Tanh, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto Asinh(const base_expr<Derived, T>& expr) {
  return unary_expr<Asinh, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto Acosh(const base_expr<Derived, T>& expr) {
  return unary_expr<Acosh, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto Atanh(const base_expr<Derived, T>& expr) {
  return unary_expr<Atanh, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto floor(const base_expr<Derived, T>& expr) {
  return unary_expr<Floor, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto ceil(const base_expr<Derived, T>& expr) {
  return unary_expr<Ceil, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto ceil(const base_expr<Derived, T>& expr) {
  return unary_expr<Ceil, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto trunc(const base_expr<Derived, T>& expr) {
  return unary_expr<Trunc, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto round(const base_expr<Derived, T>& expr) {
  return unary_expr<Round, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto erf(const base_expr<Derived, T>& expr) {
  return unary_expr<Erf, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto erfc(const base_expr<Derived, T>& expr) {
  return unary_expr<Erfc, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto tgamma(const base_expr<Derived, T>& expr) {
  return unary_expr<Tgamma, Derived, T>(expr.derived());
}

template <typename Derived, typename T>
inline auto lgamma(const base_expr<Derived, T>& expr) {
  return unary_expr<Lgamma, Derived, T>(expr.derived());
}

}  // namespace md

#endif