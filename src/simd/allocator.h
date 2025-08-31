#ifndef __MDVECTOR_ALLOCATOR_H__
#define __MDVECTOR_ALLOCATOR_H__

#include <limits>
#include <memory>

#include "simd_base.h"

namespace md {

template <class T>
class simd_allocator {
 public:
  using value_type = T;

  simd_allocator() noexcept = default;

  template <class U>
  simd_allocator(const simd_allocator<U>&) = delete;

  static constexpr size_t alignment_for() {
    if constexpr (std::is_arithmetic_v<T>) {
      return simd<T>::alignment;
    } else {
      return alignof(T);
    }
  }

  T* allocate(size_t n) {
    if (n > max_size()) {
      throw std::bad_alloc();
    }
    void* ptr =
#ifdef _WIN32
        _aligned_malloc(n * sizeof(T), alignment_for());
#else
        aligned_alloc(alignment_for(), n * sizeof(T));
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, size_t _ = 0) noexcept {
    if (p) {
#ifdef _WIN32
      _aligned_free(p);
#else
      free(p);
#endif
    }
  }

  size_t max_size() const noexcept { return std::numeric_limits<size_t>::max() / sizeof(T); }
};

template <class T>
using auto_allocator = std::conditional_t<std::is_floating_point_v<T>, simd_allocator<T>, std::allocator<T> >;

}  // namespace md

#endif  // __MDVECTOR_ALLOCATOR_H__
