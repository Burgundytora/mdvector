#ifndef __ALIGNED_ALLOCATOR_H__
#define __ALIGNED_ALLOCATOR_H__

#include <cstdlib>
#include <memory>
// 内存对齐分配器
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  static constexpr size_t alignment =
#ifdef MDVECTOR_USE_MKL
      64;  // MKL推荐对齐
#else
      32;  // AVX2基础对齐
#endif

  T* allocate(size_t n) {
    size_t aligned_size = ((n + pack_size() - 1) / pack_size()) * pack_size();
    void* ptr =
#ifdef MDVECTOR_USE_MKL
        mkl_malloc(aligned_size * sizeof(T), alignment);
#else
        aligned_alloc(alignment, aligned_size * sizeof(T));
#endif
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, size_t) {
#ifdef MDVECTOR_USE_MKL
    mkl_free(p);
#else
    free(p);
#endif
  }

 private:
  static constexpr size_t pack_size() {
    if constexpr (std::is_same_v<T, float>) {
#ifdef MDVECTOR_USE_AVX2
      return 8;  // AVX2单精度打包数
#else
      return 1;
#endif
    } else {
#ifdef MDVECTOR_USE_AVX2
      return 4;  // AVX2双精度打包数
#else
      return 1;
#endif
    }
  }
};

#endif  // __ALIGNED_ALLOCATOR_H__