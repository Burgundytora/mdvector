#ifndef __RISC_V_H__
#define __RISC_V_H__

#include "base.h"

// ======================== RISC-V Vector ========================
#include <riscv_vector.h>  // 需要支持RVV 1.0的编译器

template <>
struct simd<float> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 4;  // 假设VLEN=128位
  using type = vfloat32m1_t;

  static inline type load(const float* p) { return vle32_v_f32m1(p, pack_size); }
  static inline void store(float* p, type v) { vse32_v_f32m1(p, v, pack_size); }
  static inline type add(type a, type b) { return vfadd_vv_f32m1(a, b, pack_size); }
  static inline type sub(type a, type b) { return vfsub_vv_f32m1(a, b, pack_size); }
  static inline type mul(type a, type b) { return vfmul_vv_f32m1(a, b, pack_size); }
  static inline type div(type a, type b) { return vfdiv_vv_f32m1(a, b, pack_size); }

  static inline type mask_load(const float* p, const size_t& remaining) {
    vbool32_t mask = vmset_m_b32(remaining, pack_size);
    return vle32_v_f32m1_m(mask, vundefined_f32m1(), p, pack_size);
  }
  static inline void mask_store(float* p, const size_t& remaining, type v) {
    vbool32_t mask = vmset_m_b32(remaining, pack_size);
    vse32_v_f32m1_m(mask, p, v, pack_size);
  }

  static inline type set1(float val) { return vfmv_v_f_f32m1(val, pack_size); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 2;  // 假设VLEN=128位
  using type = vfloat64m1_t;

  static inline type load(const double* p) { return vle64_v_f64m1(p, pack_size); }
  static inline void store(double* p, type v) { vse64_v_f64m1(p, v, pack_size); }
  static inline type add(type a, type b) { return vfadd_vv_f64m1(a, b, pack_size); }
  static inline type sub(type a, type b) { return vfsub_vv_f64m1(a, b, pack_size); }
  static inline type mul(type a, type b) { return vfmul_vv_f64m1(a, b, pack_size); }
  static inline type div(type a, type b) { return vfdiv_vv_f64m1(a, b, pack_size); }

  static inline type mask_load(const double* p, const size_t& remaining) {
    vbool64_t mask = vmset_m_b64(remaining, pack_size);
    return vle64_v_f64m1_m(mask, vundefined_f64m1(), p, pack_size);
  }
  static inline void mask_store(double* p, const size_t& remaining, type v) {
    vbool64_t mask = vmset_m_b64(remaining, pack_size);
    vse64_v_f64m1_m(mask, p, v, pack_size);
  }

  static inline type set1(double val) { return vfmv_v_f_f64m1(val, pack_size); }
};
#endif  // __RISC_V_H__