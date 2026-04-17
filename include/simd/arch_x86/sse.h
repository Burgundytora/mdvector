#ifndef __MDVECTOR_SSE__
#define __MDVECTOR_SSE__

#include "../simd_base.h"
// ======================== SSE ========================
#include <emmintrin.h>  // SSE2
#include <xmmintrin.h>  // SSE

namespace md {

template <>
struct simd<float> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 4;
  using type = __m128;

  // ==================== 广播 ====================
  static inline type set1(float val) { return _mm_set1_ps(val); }

  // ==================== 读写 ====================
  static inline type load(const float* p) { return _mm_load_ps(p); }
  static inline void store(float* p, type v) { _mm_store_ps(p, v); }
  static inline type loadu(const float* p) { return _mm_loadu_ps(p); }
  static inline void storeu(float* p, type v) { _mm_storeu_ps(p, v); }

  // ==================== 掩码读写 ====================
  static inline type mask_load(const float* p, const size_t& remaining) {
    alignas(16) float tmp[4] = {0, 0, 0, 0};
    for (int i = 0; i < remaining; ++i) {
      tmp[i] = p[i];
    }
    return _mm_load_ps(tmp);
  }
  static inline void mask_store(float* p, const size_t& remaining, type v) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    for (int i = 0; i < remaining; ++i) {
      p[i] = tmp[i];
    }
  }
  static inline type mask_loadu(const float* p, const size_t& remaining) {
    alignas(16) float tmp[4] = {0, 0, 0, 0};
    for (int i = 0; i < remaining; ++i) {
      tmp[i] = p[i];
    }
    return _mm_loadu_ps(tmp);
  }
  static inline void mask_storeu(float* p, const size_t& remaining, type v) {
    alignas(16) float tmp[4];
    _mm_storeu_ps(tmp, v);
    for (int i = 0; i < remaining; ++i) {
      p[i] = tmp[i];
    }
  }

  // ==================== 四则运算 ====================
  static inline type add(type a, type b) { return _mm_add_ps(a, b); }
  static inline type sub(type a, type b) { return _mm_sub_ps(a, b); }
  static inline type mul(type a, type b) { return _mm_mul_ps(a, b); }
  static inline type div(type a, type b) { return _mm_div_ps(a, b); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 2;
  using type = __m128d;

  // ==================== 广播 ====================
  static inline type set1(double val) { return _mm_set1_pd(val); }

  // ==================== 读写 ====================
  static inline type load(const double* p) { return _mm_load_pd(p); }
  static inline void store(double* p, type v) { _mm_store_pd(p, v); }
  static inline type loadu(const double* p) { return _mm_loadu_pd(p); }
  static inline void storeu(double* p, type v) { _mm_storeu_pd(p, v); }

  // ==================== 掩码读写 ====================
  static inline type mask_load(const double* p, const size_t& remaining) {
    alignas(16) double tmp[2] = {0, 0};
    for (int i = 0; i < remaining; ++i) {
      tmp[i] = p[i];
    }
    return _mm_load_pd(tmp);
  }
  static inline void mask_store(double* p, const size_t& remaining, type v) {
    alignas(16) double tmp[2];
    _mm_store_pd(tmp, v);
    for (int i = 0; i < remaining; ++i) {
      p[i] = tmp[i];
    }
  }
  static inline type mask_loadu(const double* p, const size_t& remaining) {
    alignas(16) double tmp[2] = {0, 0};
    for (int i = 0; i < remaining; ++i) {
      tmp[i] = p[i];
    }
    return _mm_loadu_pd(tmp);
  }
  static inline void mask_storeu(double* p, const size_t& remaining, type v) {
    alignas(16) double tmp[2];
    _mm_storeu_pd(tmp, v);
    for (int i = 0; i < remaining; ++i) {
      p[i] = tmp[i];
    }
  }

  // ==================== 四则运算 ====================
  static inline type add(type a, type b) { return _mm_add_pd(a, b); }
  static inline type sub(type a, type b) { return _mm_sub_pd(a, b); }
  static inline type mul(type a, type b) { return _mm_mul_pd(a, b); }
  static inline type div(type a, type b) { return _mm_div_pd(a, b); }
};

template <>
struct simd<int> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 4;
  using type = __m128i;
  using ref_type = __m128i&;
  using const_type = const __m128i;
  using const_ref_type = const __m128i&;

  // ==================== 广播 ====================
  static inline type set1(int val) { return _mm_set1_epi32(val); }

  // ==================== 读写 ====================
  static inline type load(const int* p) { return _mm_load_si128(reinterpret_cast<const __m128i*>(p)); }
  static inline void store(int* p, const_ref_type v) { _mm_store_si128(reinterpret_cast<__m128i*>(p), v); }
  static inline type loadu(const int* p) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); }
  static inline void storeu(int* p, const_ref_type v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v); }

  // ==================== 掩码读写 ====================

  static inline type mask_load(const int* p, const size_t& remaining) {
    alignas(16) int tmp[4] = {0, 0, 0, 0};
    for (int i = 0; i < static_cast<int>(remaining); ++i) tmp[i] = p[i];
    return _mm_load_si128(reinterpret_cast<const __m128i*>(tmp));
  }
  static inline void mask_store(int* p, const size_t& remaining, const_ref_type v) {
    alignas(16) int tmp[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(tmp), v);
    for (int i = 0; i < static_cast<int>(remaining); ++i) p[i] = tmp[i];
  }

  static inline type mask_loadu(const int* p, const size_t& remaining) {
    alignas(16) int tmp[4] = {0, 0, 0, 0};
    for (int i = 0; i < static_cast<int>(remaining); ++i) tmp[i] = p[i];
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(tmp));
  }
  static inline void mask_storeu(int* p, const size_t& remaining, const_ref_type v) {
    alignas(16) int tmp[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), v);
    for (int i = 0; i < static_cast<int>(remaining); ++i) p[i] = tmp[i];
  }

  // ==================== 四则运算 ====================
  static inline type add(const_ref_type a, const_ref_type b) { return _mm_add_epi32(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return _mm_sub_epi32(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return _mm_mullo_epi32(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) {
    alignas(16) int av[4], bv[4], rv[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(av), a);
    _mm_store_si128(reinterpret_cast<__m128i*>(bv), b);
    for (int i = 0; i < 4; ++i) rv[i] = av[i] / bv[i];
    return _mm_load_si128(reinterpret_cast<const __m128i*>(rv));
  }
};

}  // namespace md

#endif  // __MDVECTOR_SSE__