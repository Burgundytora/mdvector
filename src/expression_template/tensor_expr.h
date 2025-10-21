#ifndef __MDVECTOR_TENSOR_EXPR_H__
#define __MDVECTOR_TENSOR_EXPR_H__

#include "simd/simd.h"

namespace md {

template <typename Derived, typename T>
class tensor_expr {
 public:
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  size_t used_size() const noexcept { return derived().used_size(); }

  auto extents() const noexcept { return derived().extents(); }

  template <typename Dest, typename DestPolicy>
  void eval_to(Dest* dest) const noexcept {
    const size_t n = used_size();
    size_t i = 0;
    constexpr size_t pack_size = simd<Dest>::pack_size;

    for (; i + pack_size <= n; i += pack_size) {
      auto simd_val = derived().template eval_simd<std::remove_const_t<Dest>>(i);
      DestPolicy::template store<std::remove_const_t<Dest>>(dest + i, simd_val);
    }

    const size_t remaining = n - i;
    auto simd_val = derived().template eval_simd_mask<std::remove_const_t<Dest>>(i);
    DestPolicy::template mask_store<std::remove_const_t<Dest>>(dest + i, remaining, simd_val);
  }
};

}  // namespace md

#endif  // __TENSOR_EXPR_H__