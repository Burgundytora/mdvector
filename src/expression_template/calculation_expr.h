#ifndef __MDVECTOR_CALCULATION_EXPR_H__
#define __MDVECTOR_CALCULATION_EXPR_H__

#include "scalar_expr.h"

namespace md {

template <class T, class Policy, class = void>
struct TensorScalarType {
  using type = const T&;
};

template <class T, class Policy>
struct TensorScalarType<T, Policy, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = ScalarWrapper<T, Policy>;
};

template <class T, class Policy>
using AutoType = typename TensorScalarType<T, Policy>::type;

template <class L, class R, class Policy>
class AddExpr : public TensorExpr<AddExpr<L, R, Policy>, Policy> {
  AutoType<L, Policy> lhs;
  AutoType<R, Policy> rhs;

 public:
  AddExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

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
    return simd<T>::add(l, r);
  }

  template <class T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd<T>::add(l, r);
  }
};

template <class L, class R, class Policy>
class SubExpr : public TensorExpr<SubExpr<L, R, Policy>, Policy> {
  AutoType<L, Policy> lhs;
  AutoType<R, Policy> rhs;

 public:
  SubExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

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
    return simd<T>::sub(l, r);
  }

  template <class T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd<T>::sub(l, r);
  }
};

template <class L, class R, class Policy>
class MulExpr : public TensorExpr<MulExpr<L, R, Policy>, Policy> {
  AutoType<L, Policy> lhs;
  AutoType<R, Policy> rhs;

 public:
  MulExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

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
    return simd<T>::mul(l, r);
  }

  template <class T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd<T>::mul(l, r);
  }
};

template <class L, class R, class Policy>
class DivExpr : public TensorExpr<DivExpr<L, R, Policy>, Policy> {
  AutoType<L, Policy> lhs;
  AutoType<R, Policy> rhs;

 public:
  DivExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

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
    return simd<T>::div(l, r);
  }

  template <class T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd<T>::div(l, r);
  }
};

}  // namespace md

#endif  // __MDVECTOR_CALCULATION_EXPR_H__