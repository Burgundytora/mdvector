#ifndef __MDVECTOR_X86_AVX2_H__
#define __MDVECTOR_X86_AVX2_H__

#include "simd_base.h"

// ======================== AVX2 ========================
#include <immintrin.h>

namespace md {
template <>
struct simd<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using type = __m256;
  using ref_type = __m256&;
  using const_type = const __m256;
  using const_ref_type = const __m256&;

  static inline type load(const float* p) { return _mm256_load_ps(p); }
  static inline void store(float* p, const_ref_type v) { _mm256_store_ps(p, v); }

  static inline type loadu(const float* p) { return _mm256_loadu_ps(p); }
  static inline void storeu(float* p, const_ref_type v) { _mm256_storeu_ps(p, v); }

  static inline type add(const_ref_type a, const_ref_type b) { return _mm256_add_ps(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return _mm256_sub_ps(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return _mm256_mul_ps(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) { return _mm256_div_ps(a, b); }

  static inline const __m256i mask_table[8] = {
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),        // 0
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),       // 1
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),      // 2
      _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),     // 3
      _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),    // 4
      _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),   // 5
      _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),  // 6
      _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)  // 7
  };

  static inline type mask_load(const float* p, const size_t& remaining) {
    return _mm256_maskload_ps(p, mask_table[remaining]);
  }
  static inline void mask_store(float* p, const size_t& remaining, const_ref_type v) {
    _mm256_maskstore_ps(p, mask_table[remaining], v);
  }

  static inline type mask_loadu(const float* p, const size_t& remaining) {
    if (remaining == 0) {
      return _mm256_setzero_ps();
    }
    alignas(32) float buf[8] = {0};
    for (size_t i = 0; i < remaining; ++i) {
      buf[i] = p[i];
    }
    return _mm256_load_ps(buf);
  }
  static inline void mask_storeu(float* p, const size_t& remaining, const_ref_type v) {
    if (remaining == 0) {
      return;
    }
    alignas(32) float buf[8];
    _mm256_store_ps(buf, v);
    for (size_t i = 0; i < remaining; ++i) {
      p[i] = buf[i];
    }
  }

  static inline type set1(float val) { return _mm256_set1_ps(val); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 4;
  using type = __m256d;
  using ref_type = __m256d&;
  using const_type = const __m256d;
  using const_ref_type = const __m256d&;

  static inline type load(const double* p) { return _mm256_load_pd(p); }
  static inline void store(double* p, const_ref_type v) { _mm256_store_pd(p, v); }

  static inline type loadu(const double* p) { return _mm256_loadu_pd(p); }
  static inline void storeu(double* p, const_ref_type v) { _mm256_storeu_pd(p, v); }

  static inline type add(const_ref_type a, const_ref_type b) { return _mm256_add_pd(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return _mm256_sub_pd(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return _mm256_mul_pd(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) { return _mm256_div_pd(a, b); }

  static inline const __m256i mask_table[4] = {
      _mm256_set_epi64x(0, 0, 0, 0),    // 0
      _mm256_set_epi64x(0, 0, 0, -1),   // 1
      _mm256_set_epi64x(0, 0, -1, -1),  // 2
      _mm256_set_epi64x(0, -1, -1, -1)  // 3
  };

  static inline type mask_load(const double* p, const size_t& remaining) {
    return _mm256_maskload_pd(p, mask_table[remaining]);
  }
  static inline void mask_store(double* p, const size_t& remaining, const_ref_type v) {
    _mm256_maskstore_pd(p, mask_table[remaining], v);
  }

  static inline type mask_loadu(const double* p, const size_t& remaining) {
    if (remaining == 0) {
      return _mm256_setzero_pd();
    }
    alignas(32) double buf[4] = {0};
    for (size_t i = 0; i < remaining; ++i) {
      buf[i] = p[i];
    }
    return _mm256_load_pd(buf);
  }

  static inline void mask_storeu(double* p, const size_t& remaining, const_ref_type v) {
    if (remaining == 0) {
      return;
    }
    alignas(32) double buf[4];
    _mm256_store_pd(buf, v);
    for (size_t i = 0; i < remaining; ++i) {
      p[i] = buf[i];
    }
  }

  static inline type set1(double val) { return _mm256_set1_pd(val); }
};

}  // namespace md

#endif  // __X86_AVX2_H__