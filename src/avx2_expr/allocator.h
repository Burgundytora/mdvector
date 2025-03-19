#ifndef AVX2_ALLOCATOR2_H_
#define AVX2_ALLOCATOR2_H_

#include "simd_config.h"

// ======================== 内存分配器 ========================
template <typename T>
class AlignedAllocator2 {
 public:
  using value_type = T;
  static T* allocate(size_t n) {
    void* ptr =
#ifdef _WIN32
        _aligned_malloc(n * sizeof(T), SimdConfig<T>::alignment);
#else
        aligned_alloc(SimdConfig<T>::alignment, n * sizeof(T));
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  static void deallocate(T* p, size_t _ = 0) noexcept {
    if (p) {
#ifdef _WIN32
      _aligned_free(p);
#else
      free(p);
#endif
    }
  }
};

#endif  // AVX2_ALLOCATOR_H_
