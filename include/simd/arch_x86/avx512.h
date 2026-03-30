#ifndef __MDVECTOR_AVX512__
#define __MDVECTOR_AVX512__

#include "../simd_base.h"
// ======================== AVX512 ========================
#include <immintrin.h>

namespace md {

template <>
struct simd<float> {
  static constexpr size_t alignment = 64;
  static constexpr size_t pack_size = 16;
  using type = __m512;
  using ref_type = __m512&;
  using const_type = const __m512;
  using const_ref_type = const __m512&;

  static inline type load(const float* p) { return _mm512_load_ps(p); }
  static inline void store(float* p, const_ref_type v) { _mm512_store_ps(p, v); }

  static inline type loadu(const float* p) { return _mm512_loadu_ps(p); }
  static inline void storeu(float* p, const_ref_type v) { _mm512_storeu_ps(p, v); }

  static inline type add(const_ref_type a, const_ref_type b) { return _mm512_add_ps(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return _mm512_sub_ps(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return _mm512_mul_ps(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) { return _mm512_div_ps(a, b); }

  static inline __mmask16 mask(const size_t& remaining) { return (1u << remaining) - 1; }

  static inline type mask_load(const float* p, const size_t& remaining) {
    return _mm512_maskz_load_ps(mask(remaining), p);
  }
  static inline type mask_loadu(const float* p, const size_t& remaining) {
    return _mm512_maskz_loadu_ps(mask(remaining), p);
  }
  static inline void mask_store(float* p, const size_t& remaining, const_ref_type v) {
    _mm512_mask_store_ps(p, mask(remaining), v);
  }
  static inline void mask_storeu(float* p, const size_t& remaining, const_ref_type v) {
    _mm512_mask_storeu_ps(p, mask(remaining), v);
  }

  static inline type set1(float val) { return _mm512_set1_ps(val); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 64;
  static constexpr size_t pack_size = 8;
  using type = __m512d;
  using ref_type = __m512d&;
  using const_type = const __m512d;
  using const_ref_type = const __m512d&;

  static inline type load(const double* p) { return _mm512_load_pd(p); }
  static inline void store(double* p, type v) { _mm512_store_pd(p, v); }

  static inline type loadu(const double* p) { return _mm512_loadu_pd(p); }
  static inline void storeu(double* p, type v) { _mm512_storeu_pd(p, v); }

  static inline type add(const_ref_type a, const_ref_type b) { return _mm512_add_pd(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return _mm512_sub_pd(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return _mm512_mul_pd(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) { return _mm512_div_pd(a, b); }

  static inline __mmask8 mask(const size_t& remaining) { return (1u << remaining) - 1; }

  static inline type mask_load(const double* p, const size_t& remaining) {
    return _mm512_maskz_load_pd(mask(remaining), p);
  }
  static inline type mask_loadu(const double* p, const size_t& remaining) {
    return _mm512_maskz_loadu_pd(mask(remaining), p);
  }
  static inline void mask_store(double* p, const size_t& remaining, const_ref_type v) {
    _mm512_mask_store_pd(p, mask(remaining), v);
  }
  static inline void mask_storeu(double* p, const size_t& remaining, const_ref_type v) {
    _mm512_mask_storeu_pd(p, mask(remaining), v);
  }

  static inline type set1(double val) { return _mm512_set1_pd(val); }
};

template <>
struct simd<int> {
  static constexpr size_t alignment = 64;
  static constexpr size_t pack_size = 16;
  using type = __m512i;
  using ref_type = __m512i&;
  using const_type = const __m512i;
  using const_ref_type = const __m512i&;

  static inline type load(const int* p) { return _mm512_load_si512(reinterpret_cast<const void*>(p)); }
  static inline void store(int* p, const_ref_type v) { _mm512_store_si512(reinterpret_cast<void*>(p), v); }

  static inline type loadu(const int* p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(p)); }
  static inline void storeu(int* p, const_ref_type v) { _mm512_storeu_si512(reinterpret_cast<void*>(p), v); }

  static inline type add(const_ref_type a, const_ref_type b) { return _mm512_add_epi32(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return _mm512_sub_epi32(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return _mm512_mullo_epi32(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) {
    alignas(64) int av[16], bv[16], rv[16];
    _mm512_store_si512(reinterpret_cast<void*>(av), a);
    _mm512_store_si512(reinterpret_cast<void*>(bv), b);
    for (int i = 0; i < 16; ++i) rv[i] = av[i] / bv[i];
    return _mm512_load_si512(reinterpret_cast<const void*>(rv));
  }

  static inline __mmask16 mask(const size_t& remaining) { return (1u << remaining) - 1; }

  static inline type mask_load(const int* p, const size_t& remaining) {
    return _mm512_maskz_load_epi32(mask(remaining), p);
  }
  static inline type mask_loadu(const int* p, const size_t& remaining) {
    return _mm512_maskz_loadu_epi32(mask(remaining), p);
  }
  static inline void mask_store(int* p, const size_t& remaining, const_ref_type v) {
    _mm512_mask_store_epi32(p, mask(remaining), v);
  }
  static inline void mask_storeu(int* p, const size_t& remaining, const_ref_type v) {
    _mm512_mask_storeu_epi32(p, mask(remaining), v);
  }

  static inline type set1(int val) { return _mm512_set1_epi32(val); }
};

}  // namespace md

#endif  // __MDVECTOR_AVX512__