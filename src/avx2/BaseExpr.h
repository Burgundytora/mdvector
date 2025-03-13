#ifndef AVX2_BASEEXPR_H_
#define AVX2_BASEEXPR_H_

#include "SimdConfig.h"

namespace AVX2 {

// ======================== 表达式模板基类 ========================
template <typename Derived>
class Expr {
 public:
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  size_t size() const { return derived().size(); }

  template <typename Dest>
  void eval_to(Dest* dest) const {
    const size_t n = size();
    constexpr size_t pack_size = SimdConfig<std::remove_const_t<Dest>>::pack_size;
    size_t i = 0;

    for (; i <= n - pack_size; i += pack_size) {
      auto simd_val = derived().template eval_simd<std::remove_const_t<Dest>>(i);
      if constexpr (std::is_same_v<std::remove_const_t<Dest>, float>) {
        _mm256_store_ps(dest + i, simd_val);
      } else {
        _mm256_store_pd(dest + i, simd_val);
      }
    }

    // 处理尾部元素
    if (i < n) {
      alignas(SimdConfig<std::remove_const_t<Dest>>::alignment) std::remove_const_t<Dest> temp[pack_size] = {0};
      auto simd_val = derived().template eval_simd<std::remove_const_t<Dest>>(i);
      if constexpr (std::is_same_v<std::remove_const_t<Dest>, float>) {
        _mm256_store_ps(temp, simd_val);
      } else {
        _mm256_store_pd(temp, simd_val);
      }
      std::memcpy(dest + i, temp, (n - i) * sizeof(std::remove_const_t<Dest>));
    }
  }
};

}  // namespace AVX2

#endif  // AVX2_BASEEXPR_H_
