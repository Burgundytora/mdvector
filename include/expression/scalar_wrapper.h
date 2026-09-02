#pragma once

#include "base_expr.h"

namespace md {

template <typename T>
class scalar_wrapper : public base_expr<scalar_wrapper<T>, T> {
  T value_;
  typename simd<T>::type simd_value_;

 public:
  static constexpr size_t rank_ = 0;
  using value_type = T;
  explicit scalar_wrapper(T val) : value_(val), simd_value_(simd<T>::set1(val)) {}

  scalar_wrapper(const scalar_wrapper&) = default;

  template <typename U>
  typename simd<U>::type load_simd(size_t) const {
    if constexpr (std::is_same_v<T, U>) {
      return simd_value_;
    } else {
      return simd<U>::set1(static_cast<U>(value_));
    }
  }

  template <typename U>
  typename simd<U>::type load_simd_mask(size_t) const {
    return load_simd<U>(0);
  }

  size_t used_size() const { return 1; }
  size_t size() const { return 1; }

  T operator[](size_t) const { return value_; }

  std::array<size_t, 1> extents() const { return std::array<size_t, 1>{1}; }
};

}  // namespace md
