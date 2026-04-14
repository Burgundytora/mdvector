#ifndef __MDVECTOR_SIMD__
#define __MDVECTOR_SIMD__

#include <iostream>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(_M_AMD64)
#if defined(__AVX512F__)
#include "arch_x86/avx512.h"
#elif defined(__AVX2__)
#include "arch_x86/avx2.h"
#elif defined(__SSE4_1__)
#include "arch_x86/sse.h"
#else
#include "arch_common/none.h"
#endif
#elif defined(__arm__) || defined(__aarch64__)
#include "arch_arm/arm_neon.h"
#elif defined(__riscv)
#include "arch_risc/risc_v.h"
#else
#include "arch_common/none.h"
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

// 对齐
template <typename T>
size_t get_aligned_size(size_t size) {
  if constexpr (Numeric<T>) {
    return (size % simd<T>::pack_size == 0) ? size : ((size / simd<T>::pack_size) + 1) * simd<T>::pack_size;
  } else {
    return size;
  }
}

template <typename Derived, typename T, typename Policy, bool NoMask = true>
struct simd_io_contiguous {
  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  auto load_simd_impl(size_t i) const noexcept
    requires Numeric<T>
  {
    if constexpr (NoMask) {
      // vector array
      return Policy::load(derived().template data() + i);
    } else {
      // span
      if (i + simd<T>::pack_size <= derived().template size()) {
        return Policy::load(derived().template data() + i);
      } else {
        return Policy::mask_load(derived().template data() + i, derived().template remaining_size());
      }
    }
  }

  void store_simd_impl(size_t i, typename simd<T>::const_ref_type val) noexcept {
    if constexpr (NoMask) {
      // vector array
      Policy::store(derived().template data() + i, val);
    } else {
      // span
      if (i + simd<T>::pack_size <= derived().template size()) {
        Policy::store(derived().template data() + i, val);
      } else {
        Policy::mask_store(derived().template data() + i, derived().template remaining_size(), val);
      }
    }
  }
};

}  // namespace md

#endif  // __MDVECTOR_SIMD__