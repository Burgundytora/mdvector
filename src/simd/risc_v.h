#ifndef __MDVECTOR_RISC_V_H__
#define __MDVECTOR_RISC_V_H__

#include "simd_base.h"

// ======================== RISC-V Vector ========================
#include <riscv_vector.h>  // 需要支持RVV 1.0的编译器

namespace md {

template <>
struct simd<float> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 4;
  using type = vfloat32m1_t;
  using ref_type = vfloat32m1_t&;
  using const_type = const vfloat32m1_t;
  using const_ref_type = const vfloat32m1_t&;

  static inline type load(const float* p) { return vle32_v_f32m1(p, pack_size); }
  static inline void store(float* p, const_ref_type v) { vse32_v_f32m1(p, v, pack_size); }
  static inline type add(const_ref_type a, const_ref_type b) { return vfadd_vv_f32m1(a, b, pack_size); }
  static inline type sub(const_ref_type a, const_ref_type b) { return vfsub_vv_f32m1(a, b, pack_size); }
  static inline type mul(const_ref_type a, const_ref_type b) { return vfmul_vv_f32m1(a, b, pack_size); }
  static inline type div(const_ref_type a, const_ref_type b) { return vfdiv_vv_f32m1(a, b, pack_size); }

  static inline type mask_load(const float* p, const size_t& remaining) {
    vbool32_t mask = vmset_m_b32(remaining, pack_size);
    return vle32_v_f32m1_m(mask, vundefined_f32m1(), p, pack_size);
  }
  static inline void mask_store(float* p, const size_t& remaining, const_ref_type v) {
    vbool32_t mask = vmset_m_b32(remaining, pack_size);
    vse32_v_f32m1_m(mask, p, v, pack_size);
  }

  static inline type set1(float val) { return vfmv_v_f_f32m1(val, pack_size); }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 16;
  static constexpr size_t pack_size = 2;
  using type = vfloat64m1_t;
  using ref_type = vfloat64m1_t;
  using const_type = const vfloat64m1_t;
  using const_ref_type = const vfloat64m1_t&;

  static inline type load(const double* p) { return vle64_v_f64m1(p, pack_size); }
  static inline void store(double* p, const_ref_type v) { vse64_v_f64m1(p, v, pack_size); }

  static inline type add(const_ref_type a, const_ref_type b) { return vfadd_vv_f64m1(a, b, pack_size); }
  static inline type sub(const_ref_type a, const_ref_type b) { return vfsub_vv_f64m1(a, b, pack_size); }
  static inline type mul(const_ref_type a, const_ref_type b) { return vfmul_vv_f64m1(a, b, pack_size); }
  static inline type div(const_ref_type a, const_ref_type b) { return vfdiv_vv_f64m1(a, b, pack_size); }

  static inline type mask_load(const double* p, const size_t& remaining) {
    vbool64_t mask = vmset_m_b64(remaining, pack_size);
    return vle64_v_f64m1_m(mask, vundefined_f64m1(), p, pack_size);
  }
  static inline void mask_store(double* p, const size_t& remaining, const_ref_type v) {
    vbool64_t mask = vmset_m_b64(remaining, pack_size);
    vse64_v_f64m1_m(mask, p, v, pack_size);
  }

  static inline type set1(double val) { return vfmv_v_f_f64m1(val, pack_size); }
};

}  // namespace md

#endif  // __RISC_V_H__