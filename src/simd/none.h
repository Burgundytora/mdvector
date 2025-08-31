#ifndef __MDVECTOR_NONE_SIMD_H__
#define __MDVECTOR_NONE_SIMD_H__

#include "simd_base.h"

// ======================== NO SIMD ========================

namespace md {

template <>
struct simd<float> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 1;
  using type = float;
  using ref_type = float&;
  using const_type = const float;
  using const_ref_type = const float&;

  static inline type load(const float* p) { return *p; }
  static inline void store(float* p, const_ref_type v) { *p = v; }

  static inline type loadu(const float* p) { return *p; }
  static inline void storeu(float* p, const_ref_type v) { *p = v; }

  static inline type add(const_ref_type a, const_ref_type b) { return a + b; }
  static inline type sub(const_ref_type a, const_ref_type b) { return a - b; }
  static inline type mul(const_ref_type a, const_ref_type b) { return a * b; }
  static inline type div(const_ref_type a, const_ref_type b) { return a / b; }

  static inline type mask_load(const float* p, const size_t& remaining) { return *p; }
  static inline void mask_store(float* p, const size_t& remaining, const_ref_type v) { *p = v; }

  static inline type mask_loadu(const float* p, const size_t& remaining) { return *p; }
  static inline void mask_storeu(float* p, const size_t& remaining, const_ref_type v) { *p = v; }

  static inline type set1(float val) { return val; }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 1;
  using type = double;
  using ref_type = double&;
  using const_type = const double;
  using const_ref_type = const double&;

  static inline type load(const double* p) { return *p; }
  static inline void store(double* p, const_ref_type v) { *p = v; }

  static inline type loadu(const double* p) { return *p; }
  static inline void storeu(double* p, const_ref_type v) { *p = v; }

  static inline type add(const_ref_type a, const_ref_type b) { return a + b; }
  static inline type sub(const_ref_type a, const_ref_type b) { return a - b; }
  static inline type mul(const_ref_type a, const_ref_type b) { return a * b; }
  static inline type div(const_ref_type a, const_ref_type b) { return a / b; }

  static inline type mask_load(const double* p, const size_t& remaining) { return *p; }
  static inline void mask_store(double* p, const size_t& remaining, const_ref_type v) { *p = v; }

  static inline type mask_loadu(const double* p, const size_t& remaining) { return *p; }
  static inline void mask_storeu(double* p, const size_t& remaining, const_ref_type v) { *p = v; }

  static inline type set1(type val) { return val; }
};

}  // namespace md

#endif  //__MDVECTOR_NONE_SIMD_H__