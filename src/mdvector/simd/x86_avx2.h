#ifndef __X86_AVX2_H__
#define __X86_AVX2_H__

#include "base.h"

// ======================== AVX2 ========================
#include <immintrin.h>

template <>
struct simd<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using type = __m256;

  static inline type load(const float* p) { return _mm256_load_ps(p); }
  static inline void store(float* p, type v) { _mm256_store_ps(p, v); }
  static inline type add(type a, type b) { return _mm256_add_ps(a, b); }
  static inline type sub(type a, type b) { return _mm256_sub_ps(a, b); }
  static inline type mul(type a, type b) { return _mm256_mul_ps(a, b); }
  static inline type div(type a, type b) { return _mm256_div_ps(a, b); }

  static inline const __m256i mask_table[8] = {
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),        // 0个元素（全掩码关闭）
      _mm256_set_epi32(-1, 0, 0, 0, 0, 0, 0, 0),       // 1个元素（仅最低位 e0 启用）
      _mm256_set_epi32(-1, -1, 0, 0, 0, 0, 0, 0),      // 2个元素（e0, e1 启用）
      _mm256_set_epi32(-1, -1, -1, 0, 0, 0, 0, 0),     // 3个元素（e0, e1, e2 启用）
      _mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0),    // 4个元素（e0, e1, e2, e3 启用）
      _mm256_set_epi32(-1, -1, -1, -1, -1, 0, 0, 0),   // 5个元素（e0, e1, e2, e3, e4 启用）
      _mm256_set_epi32(-1, -1, -1, -1, -1, -1, 0, 0),  // 6个元素（e0, e1, e2, e3, e4, e5 启用）
      _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, 0)  // 7个元素（e0, e1, e2, e3, e4, e5, e6 启用）
  };

  static inline type mask_load(const float* p, const size_t& remaining) {
    return _mm256_maskload_ps(p, mask_table[remaining]);
  }
  static inline void mask_store(float* p, const size_t& remaining, type v) {
    return _mm256_maskstore_ps(p, mask_table[remaining], v);
  }

  static inline type set1(float val) { return _mm256_set1_ps(val); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 4;
  using type = __m256d;

  static inline type load(const double* p) { return _mm256_load_pd(p); }
  static inline void store(double* p, type v) { _mm256_store_pd(p, v); }
  static inline type add(type a, type b) { return _mm256_add_pd(a, b); }
  static inline type sub(type a, type b) { return _mm256_sub_pd(a, b); }
  static inline type mul(type a, type b) { return _mm256_mul_pd(a, b); }
  static inline type div(type a, type b) { return _mm256_div_pd(a, b); }

  static inline const __m256i mask_table[4] = {
      _mm256_set_epi64x(0, 0, 0, 0),    // 0元素
      _mm256_set_epi64x(-1, 0, 0, 0),   // 1元素
      _mm256_set_epi64x(-1, -1, 0, 0),  // 2元素
      _mm256_set_epi64x(-1, -1, -1, 0)  // 3元素
  };

  static inline type mask_load(const double* p, const size_t& remaining) {
    return _mm256_maskload_pd(p, mask_table[remaining]);
  }
  static inline void mask_store(double* p, const size_t& remaining, type v) {
    return _mm256_maskstore_pd(p, mask_table[remaining], v);
  }

  static inline type set1(double val) { return _mm256_set1_pd(val); }
};
#endif  // __X86_AVX2_H__