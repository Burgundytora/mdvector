#pragma once

#include "../concepts/simd_concept.h"
#include "simd_arch_select.h"
#include "simd_io_policy.h"
#include "simd_op_binary.h"
#include "simd_op_unary.h"

#include <type_traits>

namespace md {

template <typename T>
inline constexpr bool is_simd_supported_v = std::is_same_v<std::remove_cv_t<T>, float> ||
                                            std::is_same_v<std::remove_cv_t<T>, double> ||
                                            std::is_same_v<std::remove_cv_t<T>, int>;

namespace detail {

// 编译期检测实现
template <typename T>
struct simd_impl_checks {
  static_assert(HasSimdTypes<simd<T>>);
  static_assert(HasSimdBroadcast<simd<T>, T>);
  static_assert(HasSimdLoadStore<simd<T>, T>);
  static_assert(HasSimdMaskLoadStore<simd<T>, T>);
  static_assert(HasSimdArithmetic<simd<T>>);
};

template struct simd_impl_checks<float>;
template struct simd_impl_checks<double>;
template struct simd_impl_checks<int>;

}  // namespace detail

// 对齐
template <typename T>
size_t get_aligned_size(size_t size) {
  if constexpr (is_simd_supported_v<T>) {
    return (size % simd<T>::pack_size == 0) ? size : ((size / simd<T>::pack_size) + 1) * simd<T>::pack_size;
  } else {
    return size;
  }
}

}  // namespace md
