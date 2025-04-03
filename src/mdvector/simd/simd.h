#ifndef __MDVECTOR_SIMD_H__
#define __MDVECTOR_SIMD_H__

#if defined(USE_AVX2)
#include "x86_avx2.h"

#elif defined(USE_AVX512)
#include "x86_avx512.h"

#elif defined(USE_SSE)
#include "x86_sse.h"

#elif defined(USE_NEON)
#include "arm_neon.h"

#elif defined(USE_RVV)
#include "risc_v.h"

#else
#include "x86_avx2.h"  // 默认avx2

#endif

// 对齐
struct AlignedPolicy {
  template <class T>
  static inline auto load(const T* ptr) {
    return simd<T>::load(ptr);
  }

  template <class T>
  static inline auto mask_load(const T* ptr, const size_t& remaining) {
    return simd<T>::mask_load(ptr, remaining);
  }

  template <class T>
  static inline void store(T* ptr, typename simd<T>::type val) {
    simd<T>::store(ptr, val);
  }

  template <class T>
  static inline void mask_store(T* ptr, const size_t& remaining, typename simd<T>::type val) {
    simd<T>::mask_store(ptr, remaining, val);
  }
};

// 非对齐
struct UnalignedPolicy {
  template <class T>
  static inline auto load(const T* ptr) {
    return simd<T>::loadu(ptr);
  }

  template <class T>
  static inline auto mask_load(const T* ptr, const size_t& remaining) {
    return simd<T>::mask_loadu(ptr, remaining);
  }

  template <class T>
  static inline void store(T* ptr, typename simd<T>::type val) {
    simd<T>::storeu(ptr, val);
  }

  template <class T>
  static inline void mask_store(T* ptr, const size_t& remaining, typename simd<T>::type val) {
    simd<T>::mask_storeu(ptr, remaining, val);
  }
};

#endif  // __SIMD_H__