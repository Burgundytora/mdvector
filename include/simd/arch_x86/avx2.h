#ifndef __MDVECTOR__AVX2__
#define __MDVECTOR__AVX2__

#include "../simd_base.h"

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
    alignas(32) float buf[8] = {0};
    for (int i = 0; i < remaining; ++i) {
      buf[i] = p[i];
    }
    return _mm256_load_ps(buf);
  }
  static inline void mask_storeu(float* p, const size_t& remaining, const_ref_type v) {
    alignas(32) float buf[8];
    _mm256_store_ps(buf, v);
    for (int i = 0; i < remaining; ++i) {
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
    alignas(32) double buf[4] = {0};
    for (int i = 0; i < remaining; ++i) {
      buf[i] = p[i];
    }
    return _mm256_load_pd(buf);
  }

  static inline void mask_storeu(double* p, const size_t& remaining, const_ref_type v) {
    alignas(32) double buf[4];
    _mm256_store_pd(buf, v);
    for (int i = 0; i < remaining; ++i) {
      p[i] = buf[i];
    }
  }

  static inline type set1(double val) { return _mm256_set1_pd(val); }
};

template <>
struct simd<int> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using type = __m256i;
  using ref_type = __m256i&;
  using const_type = const __m256i;
  using const_ref_type = const __m256i&;

  static inline type load(const int* p) { return _mm256_load_si256(reinterpret_cast<const __m256i*>(p)); }
  static inline void store(int* p, const_ref_type v) { _mm256_store_si256(reinterpret_cast<__m256i*>(p), v); }

  static inline type loadu(const int* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
  static inline void storeu(int* p, const_ref_type v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }

  static inline type add(const_ref_type a, const_ref_type b) { return _mm256_add_epi32(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return _mm256_sub_epi32(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return _mm256_mullo_epi32(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) {
    alignas(32) int av[8], bv[8], rv[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(av), a);
    _mm256_store_si256(reinterpret_cast<__m256i*>(bv), b);
    for (int i = 0; i < 8; ++i) rv[i] = av[i] / bv[i];
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(rv));
  }

  static inline const __m256i mask_table[9] = {
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
      _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),      _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
      _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),    _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
      _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),  _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
      _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1)};

  static inline type mask_load(const int* p, const size_t& remaining) {
    return _mm256_maskload_epi32(p, mask_table[remaining]);
  }
  static inline void mask_store(int* p, const size_t& remaining, const_ref_type v) {
    _mm256_maskstore_epi32(p, mask_table[remaining], v);
  }

  static inline type mask_loadu(const int* p, const size_t& remaining) {
    alignas(32) int buf[8] = {0};
    for (int i = 0; i < static_cast<int>(remaining); ++i) buf[i] = p[i];
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(buf));
  }
  static inline void mask_storeu(int* p, const size_t& remaining, const_ref_type v) {
    alignas(32) int buf[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(buf), v);
    for (int i = 0; i < static_cast<int>(remaining); ++i) p[i] = buf[i];
  }

  static inline type set1(int val) { return _mm256_set1_epi32(val); }
};

}  // namespace md

#endif  // __MDVECTOR__AVX2__