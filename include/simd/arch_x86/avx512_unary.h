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
    if constexpr (std::is_same_v<Op, Abs>) {
      return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(0x7fffffff)));
    }
    if constexpr (std::is_same_v<Op, Sqrt>) return _mm512_sqrt_ps(v);
    if constexpr (std::is_same_v<Op, RSqrt>) return _mm512_div_ps(_mm512_set1_ps(1.0F), _mm512_sqrt_ps(v));
    if constexpr (std::is_same_v<Op, Floor>) return _mm512_floor_ps(v);
    if constexpr (std::is_same_v<Op, Ceil>) return _mm512_ceil_ps(v);
    if constexpr (std::is_same_v<Op, Trunc>) {
      return _mm512_roundscale_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
    if constexpr (std::is_same_v<Op, Round>) {
      auto sign = _mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(static_cast<int>(0x80000000u)));
      auto bias = _mm512_castsi512_ps(_mm512_or_si512(sign, _mm512_castps_si512(_mm512_set1_ps(0.5F))));
      return _mm512_roundscale_ps(_mm512_add_ps(v, bias), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    if constexpr (std::is_same_v<Op, Abs>) {
      return _mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(v), _mm512_set1_epi64(0x7fffffffffffffffLL)));
    }
    if constexpr (std::is_same_v<Op, Sqrt>) return _mm512_sqrt_pd(v);
    if constexpr (std::is_same_v<Op, RSqrt>) return _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(v));
    if constexpr (std::is_same_v<Op, Floor>) return _mm512_floor_pd(v);
    if constexpr (std::is_same_v<Op, Ceil>) return _mm512_ceil_pd(v);
    if constexpr (std::is_same_v<Op, Trunc>) {
      return _mm512_roundscale_pd(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
    if constexpr (std::is_same_v<Op, Round>) {
      auto sign = _mm512_and_si512(_mm512_castpd_si512(v), _mm512_set1_epi64(0x8000000000000000ULL));
      auto bias = _mm512_castsi512_pd(_mm512_or_si512(sign, _mm512_castpd_si512(_mm512_set1_pd(0.5))));
      return _mm512_roundscale_pd(_mm512_add_pd(v, bias), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }
  } else if constexpr (std::is_same_v<T, int> && std::is_same_v<Op, Abs>) {
    return _mm512_abs_epi32(v);
  }
}

}  // namespace md::detail
