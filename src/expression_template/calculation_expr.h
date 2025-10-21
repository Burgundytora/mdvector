#ifndef __MDVECTOR_CALCULATION_EXPR_H__
#define __MDVECTOR_CALCULATION_EXPR_H__

#include "scalar_expr.h"

namespace md {

template <class T, class = void>
struct tensor_scalar_type {
  using type = const T&;
  static constexpr size_t rank_ = T::rank_;
  using layout_type = typename T::layout_type;
  static constexpr bool is_scalar = false;
};

template <class T>
struct tensor_scalar_type<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
  using type = scalar_wrapper<T>;
  static constexpr size_t rank_ = 0;      // 标量 rank 应该是 0
  using layout_type = std::layout_right;  // 标量使用默认布局
  static constexpr bool is_scalar = true;
};

template <class T>
using AutoType = typename tensor_scalar_type<T>::type;

// 推导主要操作数的布局类型 - 优先使用非标量类型的layout
template <class L, class R>
struct derived_layout_type {
  // 如果 L 不是标量，使用 L 的 layout
  static constexpr bool l_is_scalar = tensor_scalar_type<L>::is_scalar;
  static constexpr bool r_is_scalar = tensor_scalar_type<R>::is_scalar;

  using type = std::conditional_t<!l_is_scalar, typename tensor_scalar_type<L>::layout_type,
                                  std::conditional_t<!r_is_scalar, typename tensor_scalar_type<R>::layout_type,
                                                     std::layout_right  // 如果都是标量，使用默认布局
                                                     >>;
};

template <class L, class R>
using layout_type_t = typename derived_layout_type<L, R>::type;

// 获取主要操作数的 rank
template <class L, class R>
constexpr size_t derived_rank() {
  static constexpr size_t l_rank = tensor_scalar_type<L>::rank_;
  static constexpr size_t r_rank = tensor_scalar_type<R>::rank_;
  return (l_rank > r_rank) ? l_rank : r_rank;
}

template <class T, class L, class R, class Cal>
class calculation_expr : public tensor_expr<calculation_expr<T, L, R, Cal>, T> {
 public:
  using value_type = T;
  static constexpr size_t rank_ = derived_rank<L, R>();
  using layout_type = layout_type_t<L, R>;

 private:
  AutoType<L> lhs;
  AutoType<R> rhs;

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

  template <class U>
  typename simd<U>::type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<U>(i);
    auto r = rhs.template eval_simd<U>(i);
    return simd_cal<U, Cal>(l, r);
  }

  template <class U>
  typename simd<U>::type eval_simd_mask(size_t i) const {
    auto l = lhs.template eval_simd_mask<U>(i);
    auto r = rhs.template eval_simd_mask<U>(i);
    return simd_cal<U, Cal>(l, r);
  }
};

}  // namespace md

#endif  // __MDVECTOR_CALCULATION_EXPR_H__