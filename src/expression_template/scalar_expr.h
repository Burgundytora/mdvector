#ifndef __MDVECTOR_SCALAR_EXPR_H__
#define __MDVECTOR_SCALAR_EXPR_H__

#include "tensor_expr.h"

namespace md {

template <class T, class Policy>
class scalar_wrapper : public tensor_expr<scalar_wrapper<T, Policy>, Policy> {
  typename simd<T>::type simd_value_;

 public:
  explicit scalar_wrapper(T val) : simd_value_(simd<T>::set1(val)) {}

  scalar_wrapper(const scalar_wrapper &) = delete;

  template <class U>
  typename simd<U>::type eval_simd(size_t) const {
    return simd_value_;
  }

  template <class U>
  typename simd<U>::type eval_simd_mask(size_t) const {
    return simd_value_;
  }

  size_t used_size() const { return 1; }

  std::array<size_t, 1> extents() const { return std::array<size_t, 1>{1}; }
};

}  // namespace md

#endif  // __MDVECTOR_SCALAR_EXPR_H__