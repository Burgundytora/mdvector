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
    const auto mask = _mm256_cmp_ps(condition, _mm256_setzero_ps(), _CMP_NEQ_OQ);
    return _mm256_blendv_ps(false_value, true_value, mask);
  } else if constexpr (std::is_same_v<T, double>) {
    const auto mask = _mm256_cmp_pd(condition, _mm256_setzero_pd(), _CMP_NEQ_OQ);
    return _mm256_blendv_pd(false_value, true_value, mask);
  } else {
    const auto equal_zero = _mm256_cmpeq_epi32(condition, _mm256_setzero_si256());
    const auto mask = _mm256_xor_si256(equal_zero, _mm256_set1_epi32(-1));
    return _mm256_blendv_epi8(false_value, true_value, mask);
  }
}

}  // namespace md::detail
