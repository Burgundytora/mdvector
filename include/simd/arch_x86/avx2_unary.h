#pragma once

#include <immintrin.h>

namespace md::detail {

template <class T, class Op>
inline constexpr bool has_native_simd_op_unary_v =
    std::is_same_v<Op, Neg> || std::is_same_v<Op, Abs> ||
    (std::is_floating_point_v<T> &&
     (std::is_same_v<Op, Sqrt> || std::is_same_v<Op, RSqrt> || std::is_same_v<Op, Floor> || std::is_same_v<Op, Ceil> ||
      std::is_same_v<Op, Trunc> || std::is_same_v<Op, Round>));

template <class T, class Op>
inline typename simd<T>::type native_simd_op_unary(typename simd<T>::type v) {
  if constexpr (std::is_same_v<Op, Neg>) {
    return simd<T>::sub(simd<T>::set1(T{}), v);
  } else if constexpr (std::is_same_v<T, float>) {
    if constexpr (std::is_same_v<Op, Abs>) return _mm256_andnot_ps(_mm256_set1_ps(-0.0F), v);
    if constexpr (std::is_same_v<Op, Sqrt>) return _mm256_sqrt_ps(v);
    if constexpr (std::is_same_v<Op, RSqrt>) return _mm256_div_ps(_mm256_set1_ps(1.0F), _mm256_sqrt_ps(v));
    if constexpr (std::is_same_v<Op, Floor>) return _mm256_floor_ps(v);
    if constexpr (std::is_same_v<Op, Ceil>) return _mm256_ceil_ps(v);
    if constexpr (std::is_same_v<Op, Trunc>) return _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    if constexpr (std::is_same_v<Op, Round>) {
      auto bias = _mm256_or_ps(_mm256_and_ps(v, _mm256_set1_ps(-0.0F)), _mm256_set1_ps(0.5F));
      return _mm256_round_ps(_mm256_add_ps(v, bias), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    if constexpr (std::is_same_v<Op, Abs>) return _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
    if constexpr (std::is_same_v<Op, Sqrt>) return _mm256_sqrt_pd(v);
    if constexpr (std::is_same_v<Op, RSqrt>) return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(v));
    if constexpr (std::is_same_v<Op, Floor>) return _mm256_floor_pd(v);
    if constexpr (std::is_same_v<Op, Ceil>) return _mm256_ceil_pd(v);
    if constexpr (std::is_same_v<Op, Trunc>) return _mm256_round_pd(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    if constexpr (std::is_same_v<Op, Round>) {
      auto bias = _mm256_or_pd(_mm256_and_pd(v, _mm256_set1_pd(-0.0)), _mm256_set1_pd(0.5));
      return _mm256_round_pd(_mm256_add_pd(v, bias), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
  } else if constexpr (std::is_same_v<T, int> && std::is_same_v<Op, Abs>) {
    return _mm256_abs_epi32(v);
  }
}

}  // namespace md::detail
