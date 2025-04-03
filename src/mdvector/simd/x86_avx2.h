#ifndef __MDVECTOR_X86_AVX2_H__
#define __MDVECTOR_X86_AVX2_H__

#include "simd_base.h"

// ======================== AVX2 ========================
#include <immintrin.h>

template <>
struct simd<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using type = __m256;

  // 对齐操作
  static inline type load(const float* p) { return _mm256_load_ps(p); }
  static inline void store(float* p, type v) { _mm256_store_ps(p, v); }

  // 非对齐操作
  static inline type loadu(const float* p) { return _mm256_loadu_ps(p); }
  static inline void storeu(float* p, type v) { _mm256_storeu_ps(p, v); }

  // 算术运算
  static inline type add(type a, type b) { return _mm256_add_ps(a, b); }
  static inline type sub(type a, type b) { return _mm256_sub_ps(a, b); }
  static inline type mul(type a, type b) { return _mm256_mul_ps(a, b); }
  static inline type div(type a, type b) { return _mm256_div_ps(a, b); }

  // 掩码表
  static inline const __m256i mask_table[8] = {
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),        // 0个元素
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),       // 1个元素
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),      // 2个元素
      _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),     // 3个元素
      _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),    // 4个元素
      _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),   // 5个元素
      _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),  // 6个元素
      _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1)  // 7个元素
  };

  // 对齐掩码操作
  static inline type mask_load(const float* p, const size_t& remaining) {
    return _mm256_maskload_ps(p, mask_table[remaining]);
  }
  static inline void mask_store(float* p, const size_t& remaining, type v) {
    _mm256_maskstore_ps(p, mask_table[remaining], v);
  }

  // 非对齐掩码操作
  static inline type mask_loadu(const float* p, const size_t& remaining) {
    // AVX2 没有直接的 maskloadu 指令，使用条件加载
    if (remaining == 0) {
      return _mm256_setzero_ps();
    }
    alignas(32) float buf[8] = {0};
    for (size_t i = 0; i < remaining; ++i) {
      buf[i] = p[i];
    }
    return _mm256_load_ps(buf);
  }
  static inline void mask_storeu(float* p, const size_t& remaining, type v) {
    // AVX2 没有直接的 maskstoreu 指令，使用条件存储
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

  // 对齐操作
  static inline type load(const double* p) { return _mm256_load_pd(p); }
  static inline void store(double* p, type v) { _mm256_store_pd(p, v); }

  // 非对齐操作
  static inline type loadu(const double* p) { return _mm256_loadu_pd(p); }
  static inline void storeu(double* p, type v) { _mm256_storeu_pd(p, v); }

  // 算术运算
  static inline type add(type a, type b) { return _mm256_add_pd(a, b); }
  static inline type sub(type a, type b) { return _mm256_sub_pd(a, b); }
  static inline type mul(type a, type b) { return _mm256_mul_pd(a, b); }
  static inline type div(type a, type b) { return _mm256_div_pd(a, b); }

  // 掩码表
  static inline const __m256i mask_table[4] = {
      _mm256_set_epi64x(0, 0, 0, 0),    // 0元素
      _mm256_set_epi64x(0, 0, 0, -1),   // 1元素
      _mm256_set_epi64x(0, 0, -1, -1),  // 2元素
      _mm256_set_epi64x(0, -1, -1, -1)  // 3元素
  };

  // 对齐掩码操作
  static inline type mask_load(const double* p, const size_t& remaining) {
    return _mm256_maskload_pd(p, mask_table[remaining]);
  }
  static inline void mask_store(double* p, const size_t& remaining, type v) {
    _mm256_maskstore_pd(p, mask_table[remaining], v);
  }

  // 非对齐掩码操作
  static inline type mask_loadu(const double* p, const size_t& remaining) {
    // AVX2 没有直接的 maskloadu 指令，使用条件加载
    if (remaining == 0) {
      return _mm256_setzero_pd();
    }
    alignas(32) double buf[4] = {0};
    for (size_t i = 0; i < remaining; ++i) {
      buf[i] = p[i];
    }
    return _mm256_load_pd(buf);
  }
  static inline void mask_storeu(double* p, const size_t& remaining, type v) {
    // AVX2 没有直接的 maskstoreu 指令，使用条件存储
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
#endif  // __X86_AVX2_H__