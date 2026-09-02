#pragma once

namespace md::detail {

template <class T, class Op>
inline constexpr bool has_native_simd_op_binary_v = false;

template <class T, class Op>
inline typename simd<T>::type native_simd_op_binary(typename simd<T>::const_ref_type l,
                                                    typename simd<T>::const_ref_type) {
  return l;
}

}  // namespace md::detail
