#pragma once

namespace md::detail {

template <class T>
inline constexpr bool has_native_simd_select_v = false;

template <class T>
inline typename simd<T>::type native_simd_select(typename simd<T>::const_ref_type,
                                                 typename simd<T>::const_ref_type true_value,
                                                 typename simd<T>::const_ref_type) {
  return true_value;
}

}  // namespace md::detail
