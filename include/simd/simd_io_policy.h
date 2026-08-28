#pragma once

#include "simd_arch_select.h"

namespace md {

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

}  // namespace md
