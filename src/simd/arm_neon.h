#ifndef __MDVECTOR_ARM_NEON_H__
#define __MDVECTOR_ARM_NEON_H__

#include "simd_base.h"

// ======================== NEON ========================
#include <arm_neon.h>

template <>
struct simd<float> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 4;
  using type = float32x4_t;
  using ref_type = float32x4_t&;
  using const_type = const float32x4_t;
  using const_ref_type = const float32x4_t&;

  static inline type load(const float* p) { return vld1q_f32(p); }
  static inline void store(float* p, const_ref_type v) { vst1q_f32(p, v); }

  static inline type loadu(const float* p) { return vld1q_f32(p); }
  static inline void storeu(float* p, const_ref_type v) { vst1q_f32(p, v); }

  static inline type add(const_ref_type a, const_ref_type b) { return vaddq_f32(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return vsubq_f32(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return vmulq_f32(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) {
    type recp = vrecpeq_f32(b);
    recp = vmulq_f32(vrecpsq_f32(b, recp), recp);
    return vmulq_f32(a, recp);
  }

  static inline type mask_load(const float* p, const size_t& remaining) {
    static const uint32_t mask_pattern[4] = {0, 0, 0, 0};
    uint32x4_t mask = vcltq_u32(vld1q_u32(mask_pattern), vdupq_n_u32(remaining));
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(p)), mask));
  }
  static inline void mask_store(float* p, const size_t& remaining, const_ref_type v) {
    static const uint32_t mask_pattern[4] = {0, 0, 0, 0};
    uint32x4_t mask = vcltq_u32(vld1q_u32(mask_pattern), vdupq_n_u32(remaining));
    float32x4_t old_val = vld1q_f32(p);
    float32x4_t new_val = vbslq_f32(mask, v, old_val);
    vst1q_f32(p, new_val);
  }

  static inline type mask_loadu(const float* p, const size_t& remaining) {
    static const uint32_t mask_pattern[4] = {0, 0, 0, 0};
    uint32x4_t mask = vcltq_u32(vld1q_u32(mask_pattern), vdupq_n_u32(remaining));
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(p)), mask));
  }
  static inline void mask_storeu(float* p, const size_t& remaining, const_ref_type v) {
    static const uint32_t mask_pattern[4] = {0, 0, 0, 0};
    uint32x4_t mask = vcltq_u32(vld1q_u32(mask_pattern), vdupq_n_u32(remaining));
    float32x4_t old_val = vld1q_f32(p);
    float32x4_t new_val = vbslq_f32(mask, v, old_val);
    vst1q_f32(p, new_val);
  }

  static inline type set1(float val) { return vdupq_n_f32(val); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 2;
  using type = float64x2_t;
  using ref_type = float64x2_t;
  using const_type = const float64x2_t;
  using const_ref_type = const float64x2_t&;

  static inline type load(const double* p) { return vld1q_f64(p); }
  static inline void store(double* p, const_ref_type v) { vst1q_f64(p, v); }

  static inline type loadu(const double* p) { return vld1q_f64(p); }
  static inline void storeu(double* p, const_ref_type v) { vst1q_f64(p, v); }

  static inline type add(const_ref_type a, const_ref_type b) { return vaddq_f64(a, b); }
  static inline type sub(const_ref_type a, const_ref_type b) { return vsubq_f64(a, b); }
  static inline type mul(const_ref_type a, const_ref_type b) { return vmulq_f64(a, b); }
  static inline type div(const_ref_type a, const_ref_type b) {
    type recp = vrecpeq_f64(b);
    recp = vmulq_f64(vrecpsq_f64(b, recp), recp);
    return vmulq_f64(a, recp);
  }

  static inline type mask_load(const double* p, const size_t& remaining) {
    static const uint64_t mask_pattern[2] = {0, 0};
    uint64x2_t mask = vcltq_u64(vld1q_u64(mask_pattern), vdupq_n_u64(remaining));
    return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(vld1q_f64(p)), mask));
  }
  static inline void mask_store(double* p, const size_t& remaining, const_ref_type v) {
    static const uint64_t mask_pattern[2] = {0, 0};
    uint64x2_t mask = vcltq_u64(vld1q_u64(mask_pattern), vdupq_n_u64(remaining));
    float64x2_t old_val = vld1q_f64(p);
    float64x2_t new_val = vbslq_f64(mask, v, old_val);
    vst1q_f64(p, new_val);
  }

  static inline type mask_loadu(const double* p, const size_t& remaining) {
    static const uint64_t mask_pattern[2] = {0, 0};
    uint64x2_t mask = vcltq_u64(vld1q_u64(mask_pattern), vdupq_n_u64(remaining));
    return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(vld1q_f64(p)), mask));
  }
  static inline void mask_storeu(double* p, const size_t& remaining, const_ref_type v) {
    static const uint64_t mask_pattern[2] = {0, 0};
    uint64x2_t mask = vcltq_u64(vld1q_u64(mask_pattern), vdupq_n_u64(remaining));
    float64x2_t old_val = vld1q_f64(p);
    float64x2_t new_val = vbslq_f64(mask, v, old_val);
    vst1q_f64(p, new_val);
  }

  static inline type set1(double val) { return vdupq_n_f64(val); }
};
#endif  // __ARM_NEON_H__