#ifndef __MDVECTOR_CORE_EXPRESSION__
#define __MDVECTOR_CORE_EXPRESSION__

#include "../expression_template/operator_overload.h"
#include "../simd/simd_function.h"

namespace md {

// ============================================================================
// 表达式计算基类
// ============================================================================
template <typename Derived, typename T, typename SimdPolicy>
class expression_impl {
 protected:
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

 public:
  using value_type = T;

  // ============ simd IO ============
  template <typename T2>
  auto load_simd(size_t i) const noexcept
    requires Numeric<T>
  {
    return derived().template load_simd(i);
  }

  template <typename T2>
  void store_simd(size_t i, typename simd<T2>::const_ref_type val)
    requires Numeric<T>
  {
    derived().template store_simd<T2>(i, val);
  }

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

#endif