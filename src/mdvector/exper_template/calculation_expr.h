#ifndef __MDVECTOR_CALCULATION_EXPR_H__
#define __MDVECTOR_CALCULATION_EXPR_H__

#include "scalar_expr.h"

// ======================== 表达式类 ========================
template <typename L, typename R>
class AddExpr : public Expr<AddExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  AddExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  auto extents() const { return lhs.extents(); }

  template <typename T>
  typename simd<T>::type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    return simd<T>::add(l, r);
  }

  template <typename T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd<T>::add(l, r);
  }
};

template <typename L, typename R>
class SubExpr : public Expr<SubExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  SubExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  auto extents() const { return lhs.extents(); }

  template <typename T>
  typename simd<T>::type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    return simd<T>::sub(l, r);
  }

  template <typename T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd<T>::sub(l, r);
  }
};

template <typename L, typename R>
class MulExpr : public Expr<MulExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  MulExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  auto extents() const { return lhs.extents(); }

  template <typename T>
  typename simd<T>::type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    return simd<T>::mul(l, r);
  }

  template <typename T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd<T>::mul(l, r);
  }
};

template <typename L, typename R>
class DivExpr : public Expr<DivExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  DivExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  auto extents() const { return lhs.extents(); }

  template <typename T>
  typename simd<T>::type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    return simd<T>::div(l, r);
  }

  template <typename T>
  typename simd<T>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<T>(i);
    auto r = rhs.template eval_simd_mask<T>(i);
    return simd<T>::div(l, r);
  }
};

#endif  // __MDVECTOR_CALCULATION_EXPR_H__