#ifndef __MDVECTOR_ALLOCATOR_H__
#define __MDVECTOR_ALLOCATOR_H__

#include <limits>
#include <memory>

#include "simd_base.h"

namespace md {

template <class T>
class SimdAllocator {
 public:
  using value_type = T;

  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using is_always_equal = std::true_type;

  template <class U>
  struct rebind {
    using other = SimdAllocator<U>;
  };

  SimdAllocator() noexcept = default;

  template <class U>
  SimdAllocator(const SimdAllocator<U>&) noexcept {}

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

  template <class U, class... Args>
  void construct(U* p, Args&&... args) {
    ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy(U* p) {
    p->~U();
  }

  template <class U>
  bool operator==(const SimdAllocator<U>&) const noexcept {
    return true;
  }

  template <class U>
  bool operator!=(const SimdAllocator<U>&) const noexcept {
    return false;
  }
};

template <class T>
using auto_allocator = std::conditional_t<std::is_floating_point_v<T>, SimdAllocator<T>, std::allocator<T> >;

}  // namespace md

#endif  // __MDVECTOR_ALLOCATOR_H__
