#pragma once

#include <immintrin.h>

namespace md::detail {

template <class T, class Op>
inline constexpr bool has_native_simd_op_binary_v =
    std::is_same_v<Op, Equal> || std::is_same_v<Op, NotEqual> || std::is_same_v<Op, Less> ||
    std::is_same_v<Op, LessEqual> || std::is_same_v<Op, Greater> || std::is_same_v<Op, GreaterEqual> ||
    std::is_same_v<Op, LogicalAnd> || std::is_same_v<Op, LogicalOr> || std::is_same_v<Op, LogicalXor>;

template <class T, class Op>
inline typename simd<T>::type native_simd_op_binary(typename simd<T>::const_ref_type l,
                                                    typename simd<T>::const_ref_type r) {
  if constexpr (std::is_same_v<T, float>) {
    const auto zero = _mm256_setzero_ps();
    const auto truth_l = _mm256_cmp_ps(l, zero, _CMP_NEQ_UQ);
    const auto truth_r = _mm256_cmp_ps(r, zero, _CMP_NEQ_UQ);
    __m256 mask;
    if constexpr (std::is_same_v<Op, Equal>) mask = _mm256_cmp_ps(l, r, _CMP_EQ_OQ);
    if constexpr (std::is_same_v<Op, NotEqual>) mask = _mm256_cmp_ps(l, r, _CMP_NEQ_UQ);
    if constexpr (std::is_same_v<Op, Less>) mask = _mm256_cmp_ps(l, r, _CMP_LT_OQ);
    if constexpr (std::is_same_v<Op, LessEqual>) mask = _mm256_cmp_ps(l, r, _CMP_LE_OQ);
    if constexpr (std::is_same_v<Op, Greater>) mask = _mm256_cmp_ps(l, r, _CMP_GT_OQ);
    if constexpr (std::is_same_v<Op, GreaterEqual>) mask = _mm256_cmp_ps(l, r, _CMP_GE_OQ);
    if constexpr (std::is_same_v<Op, LogicalAnd>) mask = _mm256_and_ps(truth_l, truth_r);
    if constexpr (std::is_same_v<Op, LogicalOr>) mask = _mm256_or_ps(truth_l, truth_r);
    if constexpr (std::is_same_v<Op, LogicalXor>) mask = _mm256_xor_ps(truth_l, truth_r);
    return _mm256_and_ps(mask, _mm256_set1_ps(1.0F));
  } else if constexpr (std::is_same_v<T, double>) {
    const auto zero = _mm256_setzero_pd();
    const auto truth_l = _mm256_cmp_pd(l, zero, _CMP_NEQ_UQ);
    const auto truth_r = _mm256_cmp_pd(r, zero, _CMP_NEQ_UQ);
    __m256d mask;
    if constexpr (std::is_same_v<Op, Equal>) mask = _mm256_cmp_pd(l, r, _CMP_EQ_OQ);
    if constexpr (std::is_same_v<Op, NotEqual>) mask = _mm256_cmp_pd(l, r, _CMP_NEQ_UQ);
    if constexpr (std::is_same_v<Op, Less>) mask = _mm256_cmp_pd(l, r, _CMP_LT_OQ);
    if constexpr (std::is_same_v<Op, LessEqual>) mask = _mm256_cmp_pd(l, r, _CMP_LE_OQ);
    if constexpr (std::is_same_v<Op, Greater>) mask = _mm256_cmp_pd(l, r, _CMP_GT_OQ);
    if constexpr (std::is_same_v<Op, GreaterEqual>) mask = _mm256_cmp_pd(l, r, _CMP_GE_OQ);
    if constexpr (std::is_same_v<Op, LogicalAnd>) mask = _mm256_and_pd(truth_l, truth_r);
    if constexpr (std::is_same_v<Op, LogicalOr>) mask = _mm256_or_pd(truth_l, truth_r);
    if constexpr (std::is_same_v<Op, LogicalXor>) mask = _mm256_xor_pd(truth_l, truth_r);
    return _mm256_and_pd(mask, _mm256_set1_pd(1.0));
  } else {
    const auto all = _mm256_set1_epi32(-1);
    const auto truth_l = _mm256_xor_si256(_mm256_cmpeq_epi32(l, _mm256_setzero_si256()), all);
    const auto truth_r = _mm256_xor_si256(_mm256_cmpeq_epi32(r, _mm256_setzero_si256()), all);
    __m256i mask;
    if constexpr (std::is_same_v<Op, Equal>) mask = _mm256_cmpeq_epi32(l, r);
    if constexpr (std::is_same_v<Op, NotEqual>) mask = _mm256_xor_si256(_mm256_cmpeq_epi32(l, r), all);
    if constexpr (std::is_same_v<Op, Less>) mask = _mm256_cmpgt_epi32(r, l);
    if constexpr (std::is_same_v<Op, LessEqual>) mask = _mm256_xor_si256(_mm256_cmpgt_epi32(l, r), all);
    if constexpr (std::is_same_v<Op, Greater>) mask = _mm256_cmpgt_epi32(l, r);
    if constexpr (std::is_same_v<Op, GreaterEqual>) mask = _mm256_xor_si256(_mm256_cmpgt_epi32(r, l), all);
    if constexpr (std::is_same_v<Op, LogicalAnd>) mask = _mm256_and_si256(truth_l, truth_r);
    if constexpr (std::is_same_v<Op, LogicalOr>) mask = _mm256_or_si256(truth_l, truth_r);
    if constexpr (std::is_same_v<Op, LogicalXor>) mask = _mm256_xor_si256(truth_l, truth_r);
    return _mm256_and_si256(mask, _mm256_set1_epi32(1));
  }
}

}  // namespace md::detail
