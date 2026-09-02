#pragma once

#include <immintrin.h>

namespace md::detail {

template <class T>
inline constexpr bool has_native_simd_select_v = true;

template <class T>
inline typename simd<T>::type native_simd_select(typename simd<T>::const_ref_type condition,
                                                 typename simd<T>::const_ref_type true_value,
                                                 typename simd<T>::const_ref_type false_value) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm_blendv_ps(false_value, true_value, _mm_cmpneq_ps(condition, _mm_setzero_ps()));
  } else if constexpr (std::is_same_v<T, double>) {
    return _mm_blendv_pd(false_value, true_value, _mm_cmpneq_pd(condition, _mm_setzero_pd()));
  } else {
    const auto equal_zero = _mm_cmpeq_epi32(condition, _mm_setzero_si128());
    const auto mask = _mm_xor_si128(equal_zero, _mm_set1_epi32(-1));
    return _mm_blendv_epi8(false_value, true_value, mask);
  }
}

}  // namespace md::detail
