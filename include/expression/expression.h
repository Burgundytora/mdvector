#pragma once

// #include "unary_expr.h"
#include "operator.h"
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

  template <typename E>
  Derived& operator=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    expr.template eval_to<>(derived());
    return derived();
  }

  template <typename E>
  Derived& operator+=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (derived() + expr).template eval_to<>(derived());
    return derived();
  }

  template <typename E>
  Derived& operator-=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (derived() - expr).template eval_to<>(derived());
    return derived();
  }

  template <typename E>
  Derived& operator*=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (derived() * expr).template eval_to<>(derived());
    return derived();
  }

  template <typename E>
  Derived& operator/=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
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
    return derived() * static_cast<T>(-1);
  }

  auto operator+() const noexcept
    requires Numeric<T>
  {
    return derived();
  }
};

}  // namespace md
