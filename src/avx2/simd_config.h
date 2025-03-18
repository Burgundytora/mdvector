#ifndef AVX2_SIMDCONFIG_H_
#define AVX2_SIMDCONFIG_H_

#include <immintrin.h>

#ifdef _WIN32
#define FORCE_INLINE __forceinline  // MSVC强制内联宏
#else
#define FORCE_INLINE __attribute__((always_inline))  // GCC强制内联宏
#endif

// ======================== SIMD配置 ========================
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

#endif  // AVX2_SIMDCONFIG_H_