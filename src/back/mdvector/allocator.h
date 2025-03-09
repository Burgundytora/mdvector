#include <immintrin.h>
#include <memory>

#ifdef MDVECTOR_USE_MKL
#include <mkl.h>
void set_mkl_avx2_sequential_mode() {
  mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
  mkl_enable_instructions(MKL_ENABLE_AVX2);
}
void set_mkl_avx512_sequential_mode() {
  mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
  mkl_enable_instructions(MKL_ENABLE_AVX512);
}
#endif

// SIMD 配置模板
template <typename T>
struct SimdConfig;
template <>
struct SimdConfig<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using simd_type = __m256;
#ifdef MDVECTOR_USE_MKL
  static constexpr MKL_INT mkl_vsize = 8;
#endif
};
template <>
struct SimdConfig<double> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 4;
  using simd_type = __m256d;
#ifdef MDVECTOR_USE_MKL
  static constexpr MKL_INT mkl_vsize = 8;
#endif
};

// 内存对齐分配器
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  T* allocate(size_t n) {
    size_t aligned_size = ((n + SimdConfig<T>::pack_size - 1) / SimdConfig<T>::pack_size) * SimdConfig<T>::pack_size;
    // void* ptr = aligned_alloc(SimdConfig<T>::alignment, aligned_size * sizeof(T));
    void* ptr =
#ifdef MDVECTOR_USE_MKL
        mkl_malloc(aligned_size * sizeof(T), SimdConfig<T>::alignment);
#else
        aligned_alloc(SimdConfig<T>::alignment, aligned_size * sizeof(T));
#endif
    return static_cast<T*>(ptr);
  }
  // void deallocate(T* p, size_t) noexcept { free(p); }
  void deallocate(T* p, size_t) {
#ifdef MDVECTOR_USE_MKL
    mkl_free(p);
#else
    free(p);
#endif
  }
};