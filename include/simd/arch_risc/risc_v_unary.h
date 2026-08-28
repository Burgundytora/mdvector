#pragma once

namespace md::detail {

template <class T, class Op>
inline constexpr bool has_native_simd_op_unary_v =
    std::is_same_v<Op, Neg> || (std::is_floating_point_v<T> && (std::is_same_v<Op, Abs> || std::is_same_v<Op, Sqrt>));

template <class T, class Op>
inline typename simd<T>::type native_simd_op_unary(typename simd<T>::type v) {
  if constexpr (std::is_same_v<Op, Neg>) {
    return simd<T>::sub(simd<T>::set1(T{}), v);
  } else if constexpr (std::is_same_v<T, float>) {
    if constexpr (std::is_same_v<Op, Abs>) return vfabs_v_f32m1(v, simd<T>::pack_size);
    if constexpr (std::is_same_v<Op, Sqrt>) return vfsqrt_v_f32m1(v, simd<T>::pack_size);
  } else if constexpr (std::is_same_v<T, double>) {
    if constexpr (std::is_same_v<Op, Abs>) return vfabs_v_f64m1(v, simd<T>::pack_size);
    if constexpr (std::is_same_v<Op, Sqrt>) return vfsqrt_v_f64m1(v, simd<T>::pack_size);
  }
}

}  // namespace md::detail
