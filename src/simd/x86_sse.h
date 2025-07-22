#ifndef __MDVECTOR_X86_SSE_H__
#define __MDVECTOR_X86_SSE_H__

#include "simd_base.h"

// ======================== SSE ========================
#include <emmintrin.h>  // SSE2
#include <xmmintrin.h>  // SSE

namespace md {

template <>
struct simd<float> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 4;
  using type = __m128;

  // 对齐操作
  static inline type load(const float* p) { return _mm_load_ps(p); }
  static inline void store(float* p, type v) { _mm_store_ps(p, v); }

  // 非对齐操作
  static inline type loadu(const float* p) { return _mm_loadu_ps(p); }
  static inline void storeu(float* p, type v) { _mm_storeu_ps(p, v); }

  // 算术运算
  static inline type add(type a, type b) { return _mm_add_ps(a, b); }
  static inline type sub(type a, type b) { return _mm_sub_ps(a, b); }
  static inline type mul(type a, type b) { return _mm_mul_ps(a, b); }
  static inline type div(type a, type b) { return _mm_div_ps(a, b); }

  // 对齐掩码操作（SSE没有原生支持，使用临时缓冲区）
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

  // 非对齐掩码操作
  static inline type mask_loadu(const float* p, const size_t& remaining) {
    alignas(16) float tmp[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < remaining; ++i) tmp[i] = p[i];
    return _mm_loadu_ps(tmp);  // 使用loadu保证非对齐安全
  }
  static inline void mask_storeu(float* p, const size_t& remaining, type v) {
    alignas(16) float tmp[4];
    _mm_storeu_ps(tmp, v);  // 使用storeu保证非对齐安全
    for (size_t i = 0; i < remaining; ++i) p[i] = tmp[i];
  }

  static inline type set1(float val) { return _mm_set1_ps(val); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 2;
  using type = __m128d;

  // 对齐操作
  static inline type load(const double* p) { return _mm_load_pd(p); }
  static inline void store(double* p, type v) { _mm_store_pd(p, v); }

  // 非对齐操作
  static inline type loadu(const double* p) { return _mm_loadu_pd(p); }
  static inline void storeu(double* p, type v) { _mm_storeu_pd(p, v); }

  // 算术运算
  static inline type add(type a, type b) { return _mm_add_pd(a, b); }
  static inline type sub(type a, type b) { return _mm_sub_pd(a, b); }
  static inline type mul(type a, type b) { return _mm_mul_pd(a, b); }
  static inline type div(type a, type b) { return _mm_div_pd(a, b); }

  // 对齐掩码操作
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

  // 非对齐掩码操作
  static inline type mask_loadu(const double* p, const size_t& remaining) {
    alignas(16) double tmp[2] = {0, 0};
    for (size_t i = 0; i < remaining; ++i) tmp[i] = p[i];
    return _mm_loadu_pd(tmp);
  }
  static inline void mask_storeu(double* p, const size_t& remaining, type v) {
    alignas(16) double tmp[2];
    _mm_storeu_pd(tmp, v);
    for (size_t i = 0; i < remaining; ++i) p[i] = tmp[i];
  }

  static inline type set1(double val) { return _mm_set1_pd(val); }
};

}  // namespace md

#endif  // __X86_SSE_H__