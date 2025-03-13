#ifndef AVX2_SIMDCONFIG_H_
#define AVX2_SIMDCONFIG_H_

#include <immintrin.h>

namespace AVX2 {

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

}  // namespace AVX2

#endif  // AVX2_SIMDCONFIG_H_