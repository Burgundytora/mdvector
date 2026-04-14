#ifndef __MDVECTOR_EXPRESSION__
#define __MDVECTOR_EXPRESSION__

#include "operator_overload.h"
#include "../simd/simd.h"

namespace md {

// ============================================================================
// 表达式计算基类
// ============================================================================
template <typename Derived, typename T, typename SimdPolicy>
class expression_impl {
 protected:
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

 public:
  using value_type = T;
  // ============ SIMD IO ============

  // template <typename T>
  auto load_simd(size_t i) const noexcept
    requires Numeric<T>
  {
    return derived().template load_simd_impl(i);
  }

  // template <typename T2>
  void store_simd(size_t i, typename simd<T>::const_ref_type val)
    requires Numeric<T>
  {
    derived().template store_simd_impl(i, val);
  }

  size_t used_size() const noexcept { return derived().used_size(); }

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
    (derived() + scalar).template eval_to<>(*this);
    return derived();
  }

  Derived& operator-=(T scalar) noexcept
    requires Numeric<T>
  {
    (derived() - scalar).template eval_to<>(*this);
    return derived();
  }

  Derived& operator*=(T scalar) noexcept
    requires Numeric<T>
  {
    (derived() * scalar).template eval_to<>(*this);
    return derived();
  }

  Derived& operator/=(T scalar) noexcept
    requires Numeric<T>
  {
    (derived() / scalar).template eval_to<>(*this);
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

#endif  // __MDVECTOR_EXPRESSION__