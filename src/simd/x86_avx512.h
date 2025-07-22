#ifndef __MDVECTOR_X86_AVX512_H__
#define __MDVECTOR_X86_AVX512_H__

#include "simd_base.h"

// ======================== AVX512 ========================

#include <immintrin.h>

namespace md {

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

  static inline __mmask16 mask(const size_t& remaining) { return (1u << remaining) - 1; }

  static inline type mask_load(const float* p, const size_t& remaining) {
    return _mm512_maskz_load_ps(mask(remaining), p);
  }
  static inline type mask_loadu(const float* p, const size_t& remaining) {
    return _mm512_maskz_loadu_ps(mask(remaining), p);
  }
  static inline void mask_store(float* p, const size_t& remaining, type v) {
    _mm512_mask_store_ps(p, mask(remaining), v);
  }
  static inline void mask_storeu(float* p, const size_t& remaining, type v) {
    _mm512_mask_storeu_ps(p, mask(remaining), v);
  }

  static inline type set1(float val) { return _mm512_set1_ps(val); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 64;
  static constexpr size_t pack_size = 8;
  using type = __m512d;

  static inline type add(type a, type b) { return _mm512_add_pd(a, b); }
  static inline type sub(type a, type b) { return _mm512_sub_pd(a, b); }
  static inline type mul(type a, type b) { return _mm512_mul_pd(a, b); }
  static inline type div(type a, type b) { return _mm512_div_pd(a, b); }

  static inline type load(const double* p) { return _mm512_load_pd(p); }
  static inline void store(double* p, type v) { _mm512_store_pd(p, v); }
  static inline type loadu(const double* p) { return _mm512_loadu_pd(p); }
  static inline void storeu(double* p, type v) { _mm512_storeu_pd(p, v); }

  static inline __mmask8 mask(const size_t& remaining) { return (1u << remaining) - 1; }

  static inline type mask_load(const double* p, const size_t& remaining) {
    return _mm512_maskz_load_pd(mask(remaining), p);
  }
  static inline type mask_loadu(const double* p, const size_t& remaining) {
    return _mm512_maskz_loadu_pd(mask(remaining), p);
  }
  static inline void mask_store(double* p, const size_t& remaining, type v) {
    _mm512_mask_store_pd(p, mask(remaining), v);
  }
  static inline void mask_storeu(double* p, const size_t& remaining, type v) {
    _mm512_mask_storeu_pd(p, mask(remaining), v);
  }

  static inline type set1(double val) { return _mm512_set1_pd(val); }
};

}  // namespace md

#endif  // __X86_AVX512_H__