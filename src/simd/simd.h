#ifndef __MDVECTOR_SIMD_H__
#define __MDVECTOR_SIMD_H__

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(_M_AMD64)
#if defined(__AVX512F__)
#include "x86_avx512.h"
#elif defined(__AVX2__)
#include "x86_avx2.h"
#elif defined(__SSE4_1__)
#include "x86_sse.h"
#else
#include "none.h"
#endif
#elif defined(__arm__) || defined(__aarch64__)
#include "arm_neon.h"
#elif defined(__riscv)
#include "risc_v.h"
#else
#include "none.h"
#endif

namespace md {

void print_simd_type() {
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(_M_AMD64)
#if defined(__AVX512F__)
  std::cout << "x86 avx512...\n";
#elif defined(__AVX2__)
  std::cout << "x86 avx2...\n";
#elif defined(__SSE4_1__)
  std::cout << "x86 sse...\n";
#else
  std::cout << "x86 none...\n";
#endif
#elif defined(__arm__) || defined(__aarch64__)
  std::cout << "arm neon...\n";
#elif defined(__riscv)
  std::cout << "riscv...\n";
#else
  std::cout << "none...\n";
#endif
}

// 对齐
struct aligned_policy {
  template <class T>
  static inline auto load(const T* ptr) {
    return simd<T>::load(ptr);
  }

  template <class T>
  static inline auto mask_load(const T* ptr, const size_t& remaining) {
    return simd<T>::mask_load(ptr, remaining);
  }

  template <class T>
  static inline void store(T* ptr, typename simd<T>::const_ref_type val) {
    simd<T>::store(ptr, val);
  }

  template <class T>
  static inline void mask_store(T* ptr, const size_t& remaining, typename simd<T>::const_ref_type val) {
    simd<T>::mask_store(ptr, remaining, val);
  }
};

// 非对齐
struct unaligned_policy {
  template <class T>
  static inline auto load(const T* ptr) {
    return simd<T>::loadu(ptr);
  }

  template <class T>
  static inline auto mask_load(const T* ptr, const size_t& remaining) {
    return simd<T>::mask_loadu(ptr, remaining);
  }

  template <class T>
  static inline void store(T* ptr, typename simd<T>::const_ref_type val) {
    simd<T>::storeu(ptr, val);
  }

  template <class T>
  static inline void mask_store(T* ptr, const size_t& remaining, typename simd<T>::const_ref_type val) {
    simd<T>::mask_storeu(ptr, remaining, val);
  }
};

struct Add;
struct Sub;
struct Mul;
struct Div;

template <class T, class Cal>
static inline typename simd<T>::type simd_cal(typename simd<T>::const_ref_type l, typename simd<T>::const_ref_type r) {
  if constexpr (std::is_same_v<Cal, Add>) {
    return simd<T>::add(l, r);
  } else if constexpr (std::is_same_v<Cal, Sub>) {
    return simd<T>::sub(l, r);
  } else if constexpr (std::is_same_v<Cal, Mul>) {
    return simd<T>::mul(l, r);
  } else if constexpr (std::is_same_v<Cal, Div>) {
    return simd<T>::div(l, r);
  } else {
    static_assert(false, "simd_cal<T, Cal>, Cal must be Add/Sub/Mul/Div !");
  }
}

}  // namespace md

#endif  // __SIMD_H__