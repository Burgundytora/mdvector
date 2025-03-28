#ifndef __X86_AVX512_H__
#define __X86_AVX512_H__

#include "base.h"

// ======================== AVX512 ========================

#include <immintrin.h>

template <>
struct simd<float> {
  static constexpr size_t alignment = 64;
  static constexpr size_t pack_size = 16;
  using type = __m512;

  static inline type load(const float* p) { return _mm512_load_ps(p); }
  static inline void store(float* p, type v) { _mm512_store_ps(p, v); }
  static inline type add(type a, type b) { return _mm512_add_ps(a, b); }
  static inline type sub(type a, type b) { return _mm512_sub_ps(a, b); }
  static inline type mul(type a, type b) { return _mm512_mul_ps(a, b); }
  static inline type div(type a, type b) { return _mm512_div_ps(a, b); }

  static inline type mask_load(const float* p, const size_t& remaining) {
    __mmask16 mask = (1u << remaining) - 1;
    return _mm512_maskz_load_ps(mask, p);
  }
  static inline void mask_store(float* p, const size_t& remaining, type v) {
    __mmask16 mask = (1u << remaining) - 1;
    _mm512_mask_store_ps(p, mask, v);
  }

  static inline type set1(float val) { return _mm512_set1_ps(val); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 64;
  static constexpr size_t pack_size = 8;
  using type = __m512d;

  static inline type load(const double* p) { return _mm512_load_pd(p); }
  static inline void store(double* p, type v) { _mm512_store_pd(p, v); }
  static inline type add(type a, type b) { return _mm512_add_pd(a, b); }
  static inline type sub(type a, type b) { return _mm512_sub_pd(a, b); }
  static inline type mul(type a, type b) { return _mm512_mul_pd(a, b); }
  static inline type div(type a, type b) { return _mm512_div_pd(a, b); }

  static inline type mask_load(const double* p, const size_t& remaining) {
    __mmask8 mask = (1u << remaining) - 1;
    return _mm512_maskz_load_pd(mask, p);
  }
  static inline void mask_store(double* p, const size_t& remaining, type v) {
    __mmask8 mask = (1u << remaining) - 1;
    _mm512_mask_store_pd(p, mask, v);
  }

  static inline type set1(double val) { return _mm512_set1_pd(val); }
};
#endif  // __X86_AVX512_H__