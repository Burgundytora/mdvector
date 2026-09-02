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
    const auto mask = _mm512_cmp_ps_mask(condition, _mm512_setzero_ps(), _CMP_NEQ_OQ);
    return _mm512_mask_blend_ps(mask, false_value, true_value);
  } else if constexpr (std::is_same_v<T, double>) {
    const auto mask = _mm512_cmp_pd_mask(condition, _mm512_setzero_pd(), _CMP_NEQ_OQ);
    return _mm512_mask_blend_pd(mask, false_value, true_value);
  } else {
    const auto mask = _mm512_cmpneq_epi32_mask(condition, _mm512_setzero_si512());
    return _mm512_mask_blend_epi32(mask, false_value, true_value);
  }
}

}  // namespace md::detail
