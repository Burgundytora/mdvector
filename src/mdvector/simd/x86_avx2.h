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
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),         // 0元素
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),        // 1元素
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),       // 2元素
      _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),     // 3元素
      _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),    // 4元素
      _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),   // 5元素
      _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),   // 6元素
      _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),  // 7元素
  };

  static inline type mask_load(const float* p, const size_t& remaining) {
    return _mm256_maskload_ps(p, mask_table[remaining]);
  }
  static inline void mask_store(float* p, const size_t& remaining, type v) {
    return _mm256_maskstore_ps(p, mask_table[remaining], v);
  }
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
      _mm256_set_epi64x(0, 0, 0, -1),   // 1元素
      _mm256_set_epi64x(0, 0, -1, -1),  // 2元素
      _mm256_set_epi64x(0, -1, -1, -1)  // 3元素
  };

  static inline type mask_load(const double* p, const size_t& remaining) {
    return _mm256_maskload_pd(p, mask_table[remaining]);
  }
  static inline void mask_store(double* p, const size_t& remaining, type v) {
    return _mm256_maskstore_pd(p, mask_table[remaining], v);
  }
};
#endif  // __X86_AVX2_H__