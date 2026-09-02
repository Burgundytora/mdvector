#pragma once

#include "operator.h"
#include "conditional_expr.h"
#include "unary_expr.h"
#include "../simd/simd.h"

namespace md {

// ============================================================================
// 表达式计算基类
// ============================================================================
template <typename Derived, typename T, typename SimdPolicy>
class expression {
 protected:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

 public:
  using value_type = T;

  // ============ 表达式模板赋值 ============

  template <typename E, typename U>
  Derived& operator=(const base_expr<E, U>& expr)
    requires(Numeric<T> && std::convertible_to<U, T>)
  {
    expr.template eval_to<>(derived());
    return derived();
  }

  template <typename E, typename U>
  Derived& operator+=(const base_expr<E, U>& expr)
    requires(Numeric<T> && std::convertible_to<U, T>)
  {
    (derived() + expr).template eval_to<>(derived());
    return derived();
  }

  template <typename E, typename U>
  Derived& operator-=(const base_expr<E, U>& expr)
    requires(Numeric<T> && std::convertible_to<U, T>)
  {
    (derived() - expr).template eval_to<>(derived());
    return derived();
  }

  template <typename E, typename U>
  Derived& operator*=(const base_expr<E, U>& expr)
    requires(Numeric<T> && std::convertible_to<U, T>)
  {
    (derived() * expr).template eval_to<>(derived());
    return derived();
  }

  template <typename E, typename U>
  Derived& operator/=(const base_expr<E, U>& expr)
    requires(Numeric<T> && std::convertible_to<U, T>)
  {
    (derived() / expr).template eval_to<>(derived());
    return derived();
  }

  // ============ 标量运算 ============

  Derived& operator+=(T scalar) noexcept
    requires Numeric<T>
  {
    (derived() + scalar).template eval_to<>(derived());
    return derived();
  }

  Derived& operator-=(T scalar) noexcept
    requires Numeric<T>
  {
    (derived() - scalar).template eval_to<>(derived());
    return derived();
  }

  Derived& operator*=(T scalar) noexcept
    requires Numeric<T>
  {
    (derived() * scalar).template eval_to<>(derived());
    return derived();
  }

  Derived& operator/=(T scalar) noexcept
    requires Numeric<T>
  {
    (derived() / scalar).template eval_to<>(derived());
    return derived();
  }

  // ============ 一元运算 ============

  auto operator-() const noexcept
    requires Numeric<T>
  {
    return unary_expr<Neg, Derived, T>(derived());
  }

  auto operator+() const noexcept
    requires Numeric<T>
  {
    return derived();
  }
};

}  // namespace md
