#ifndef __MDVECTOR_NONE_SIMD_H__
#define __MDVECTOR_NONE_SIMD_H__

#include "simd_base.h"

// 不使用simd

namespace md {

template <>
struct simd<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 1;
  using type = float;

  // 对齐操作
  static inline type load(const type* p) { return *p; }
  static inline void store(type* p, type v) { *p = v; }

  // 非对齐操作
  static inline type loadu(const type* p) { return *p; }
  static inline void storeu(type* p, type v) { *p = v; }

  // 算术运算
  static inline type add(type a, type b) { return a + b; }
  static inline type sub(type a, type b) { return a - b; }
  static inline type mul(type a, type b) { return a * b; }
  static inline type div(type a, type b) { return a / b; }

  // 对齐掩码操作
  static inline type mask_load(const type* p, const size_t& remaining) { return *p; }
  static inline void mask_store(type* p, const size_t& remaining, type v) { *p = v; }

  // 非对齐掩码操作
  static inline type mask_loadu(const type* p, const size_t& remaining) { return *p; }
  static inline void mask_storeu(type* p, const size_t& remaining, type v) { *p = v; }

  static inline type set1(type val) { return val; }
};

template <>
struct simd<double> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 1;
  using type = double;

  // 对齐操作
  static inline type load(const type* p) { return *p; }
  static inline void store(type* p, type v) { *p = v; }

  // 非对齐操作
  static inline type loadu(const type* p) { return *p; }
  static inline void storeu(type* p, type v) { *p = v; }

  // 算术运算
  static inline type add(type a, type b) { return a + b; }
  static inline type sub(type a, type b) { return a - b; }
  static inline type mul(type a, type b) { return a * b; }
  static inline type div(type a, type b) { return a / b; }

  // 对齐掩码操作
  static inline type mask_load(const type* p, const size_t& remaining) { return *p; }
  static inline void mask_store(type* p, const size_t& remaining, type v) { *p = v; }

  // 非对齐掩码操作
  static inline type mask_loadu(const type* p, const size_t& remaining) { return *p; }
  static inline void mask_storeu(type* p, const size_t& remaining, type v) { *p = v; }

  static inline type set1(type val) { return val; }
};

}  // namespace md

#endif  //__MDVECTOR_NONE_SIMD_H__