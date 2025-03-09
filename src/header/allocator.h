#ifndef HEADER_ALLOCATOR_H_
#define HEADER_ALLOCATOR_H_

#include <immintrin.h>

// SIMD内存对齐配置模板
template <typename T>
struct SimdConfig;
template <>
struct SimdConfig<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using simd_type = __m256;
};
template <>
struct SimdConfig<double> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 4;
  using simd_type = __m256d;
};

template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  T* allocate(size_t n) {
    size_t aligned_size = ((n + SimdConfig<T>::pack_size - 1) / SimdConfig<T>::pack_size) * SimdConfig<T>::pack_size;
    aligned_size = std::max(aligned_size, SimdConfig<T>::pack_size);
    void* ptr =
#ifdef _WIN32
        _aligned_malloc(aligned_size * sizeof(T), SimdConfig<T>::alignment);
#else
        aligned_alloc(SimdConfig<T>::alignment, aligned_size * sizeof(T));
#endif
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p) noexcept {
    if (p) {
#ifdef _WIN32
      _aligned_free(p);
#else
      free(p);
#endif
    }
  }
};

#endif  // HEADER_ALLOCATOR_H_