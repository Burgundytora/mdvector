#ifndef __MDVECTOR_CALCULATION_EXPR_H__
#define __MDVECTOR_CALCULATION_EXPR_H__

#include "scalar_expr.h"

namespace md {

template <class T, class Policy, class = void>
struct tensor_scalar_type {
  using type = const T&;
};

template <class T, class Policy>
struct tensor_scalar_type<T, Policy, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = scalar_wrapper<T, Policy>;
};

template <class T, class Policy>
using AutoType = typename tensor_scalar_type<T, Policy>::type;

template <class L, class R, class Policy, class Cal>
class calculation_expr : public tensor_expr<calculation_expr<L, R, Policy, Cal>, Policy> {
  AutoType<L, Policy> lhs;
  AutoType<R, Policy> rhs;

 public:
  calculation_expr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t used_size() const {
    if constexpr (std::is_arithmetic_v<R>) {
      return lhs.used_size();
    } else {
      return rhs.used_size();
    }
  }

  auto extents() const {
    if constexpr (std::is_arithmetic_v<R>) {
      return lhs.extents();
    } else {
      return rhs.extents();
    }
  }

  template <class T>
  typename simd<T>::type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    return simd_cal<T, Cal>(l, r);
  }

  template <class T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd_cal<T, Cal>(l, r);
  }
};

}  // namespace md

#endif  // __MDVECTOR_CALCULATION_EXPR_H__