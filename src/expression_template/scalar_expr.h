#ifndef __MDVECTOR_SCALAR_EXPR__
#define __MDVECTOR_SCALAR_EXPR__

#include "base_expr.h"

namespace md {

template <typename T>
class scalar_wrapper : public base_expr<scalar_wrapper<T>, T> {
  typename simd<T>::type simd_value_;
  static constexpr size_t rank_ = 0;
  using value_type = T;

 public:
  explicit scalar_wrapper(T val) : simd_value_(simd<T>::set1(val)) {}

  scalar_wrapper(const scalar_wrapper &) = delete;

  template <typename U>
  typename simd<U>::type load_simd(size_t) const {
    return simd_value_;
  }

  template <typename U>
  typename simd<U>::type load_simd_mask(size_t) const {
    return simd_value_;
  }

  size_t used_size() const { return 1; }

  std::array<size_t, 1> extents() const { return std::array<size_t, 1>{1}; }
};

}  // namespace md

#endif  // __MDVECTOR_SCALAR_EXPR__