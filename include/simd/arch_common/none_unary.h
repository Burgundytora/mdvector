#pragma once

namespace md::detail {

template <class T, class Op>
inline constexpr bool has_native_simd_op_unary_v = std::is_same_v<Op, Neg>;

template <class T, class Op>
inline typename simd<T>::type native_simd_op_unary(typename simd<T>::type v) {
  return simd<T>::sub(simd<T>::set1(T{}), v);
}

}  // namespace md::detail
