#ifndef __X86_SSE_H__
#define __X86_SSE_H__

#include "base.h"

// ======================== SSE ========================

#include <emmintrin.h>  // SSE2
#include <xmmintrin.h>  // SSE

template <>
struct simd<float> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 4;
  using type = __m128;

  static inline type load(const float* p) { return _mm_load_ps(p); }
  static inline void store(float* p, type v) { _mm_store_ps(p, v); }
  static inline type add(type a, type b) { return _mm_add_ps(a, b); }
  static inline type sub(type a, type b) { return _mm_sub_ps(a, b); }
  static inline type mul(type a, type b) { return _mm_mul_ps(a, b); }
  static inline type div(type a, type b) { return _mm_div_ps(a, b); }

  // SSE 没有直接掩码加载指令，通过混合操作模拟
  static inline type mask_load(const float* p, const size_t& remaining) {
    alignas(16) float tmp[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < remaining; ++i) tmp[i] = p[i];
    return _mm_load_ps(tmp);
  }
  static inline void mask_store(float* p, const size_t& remaining, type v) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    for (size_t i = 0; i < remaining; ++i) p[i] = tmp[i];
  }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 2;
  using type = __m128d;

  static inline type load(const double* p) { return _mm_load_pd(p); }
  static inline void store(double* p, type v) { _mm_store_pd(p, v); }
  static inline type add(type a, type b) { return _mm_add_pd(a, b); }
  static inline type sub(type a, type b) { return _mm_sub_pd(a, b); }
  static inline type mul(type a, type b) { return _mm_mul_pd(a, b); }
  static inline type div(type a, type b) { return _mm_div_pd(a, b); }

  static inline type mask_load(const double* p, const size_t& remaining) {
    alignas(16) double tmp[2] = {0, 0};
    for (size_t i = 0; i < remaining; ++i) tmp[i] = p[i];
    return _mm_load_pd(tmp);
  }
  static inline void mask_store(double* p, const size_t& remaining, type v) {
    alignas(16) double tmp[2];
    _mm_store_pd(tmp, v);
    for (size_t i = 0; i < remaining; ++i) p[i] = tmp[i];
  }
};
#endif  // __X86_SSE_H__