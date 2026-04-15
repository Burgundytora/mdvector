#ifndef __MDVECTOR_OP_BINATY__
#define __MDVECTOR_OP_BINATY__

#include "simd_arch_select.h"

namespace md {

// 四则运算
struct Add;
struct Sub;
struct Mul;
struct Div;

// TODO: 二元函数
struct Exp;
struct Pow;
struct Hypot;
struct Fmod;

template <class T, class Cal>
static inline typename simd<T>::type simd_op_binary(typename simd<T>::const_ref_type l,
                                                    typename simd<T>::const_ref_type r) {
  if constexpr (std::is_same_v<Cal, Add>) {
    return simd<T>::add(l, r);
  } else if constexpr (std::is_same_v<Cal, Sub>) {
    return simd<T>::sub(l, r);
  } else if constexpr (std::is_same_v<Cal, Mul>) {
    return simd<T>::mul(l, r);
  } else if constexpr (std::is_same_v<Cal, Div>) {
    return simd<T>::div(l, r);
  } else {
    static_assert(false, "simd_op_binary only support Add/Sub/Mul/Div now !");
  }
}

}  // namespace md

#endif  // __MDVECTOR_OP_BINATY__