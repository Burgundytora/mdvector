#ifndef __MDVECTOR_TENSOR_EXPR_H__
#define __MDVECTOR_TENSOR_EXPR_H__

#include "simd/simd.h"

namespace md {

template <class Derived, class Policy>
class tensor_expr {
 public:
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

  size_t used_size() const noexcept { return derived().used_size(); }

  auto extents() const noexcept { return derived().extents(); }

  auto eval_simd(size_t i) const noexcept { return static_cast<const Derived&>(*this).eval_simd(i); }

  template <class Dest>
  void eval_to(Dest* dest) const noexcept {
    const size_t n = used_size();
    constexpr size_t pack_size = simd<Dest>::pack_size;
    size_t i = 0;

    for (; i + pack_size <= n; i += pack_size) {
      auto simd_val = derived().template eval_simd<std::remove_const_t<Dest>>(i);
      Policy::template store<std::remove_const_t<Dest>>(dest + i, simd_val);
    }

    const size_t remaining = n - i;
    auto simd_val = derived().template eval_simd_mask<std::remove_const_t<Dest>>(i);
    Policy::template mask_store<std::remove_const_t<Dest>>(dest + i, remaining, simd_val);
  }
};

}  // namespace md

#endif  // __TENSOR_EXPR_H__