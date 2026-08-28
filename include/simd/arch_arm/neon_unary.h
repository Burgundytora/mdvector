#pragma once

namespace md::detail {

template <class T, class Op>
inline constexpr bool has_native_simd_op_unary_v =
    std::is_same_v<Op, Neg> || std::is_same_v<Op, Abs>
#if defined(__aarch64__)
    || (std::is_floating_point_v<T> &&
        (std::is_same_v<Op, Sqrt> || std::is_same_v<Op, RSqrt> || std::is_same_v<Op, Floor> ||
         std::is_same_v<Op, Ceil> || std::is_same_v<Op, Trunc> || std::is_same_v<Op, Round>))
#endif
    ;

template <class T, class Op>
inline typename simd<T>::type native_simd_op_unary(typename simd<T>::type v) {
  if constexpr (std::is_same_v<Op, Neg>) {
    return simd<T>::sub(simd<T>::set1(T{}), v);
  } else if constexpr (std::is_same_v<T, float>) {
    if constexpr (std::is_same_v<Op, Abs>) return vabsq_f32(v);
#if defined(__aarch64__)
    if constexpr (std::is_same_v<Op, Sqrt>) return vsqrtq_f32(v);
    if constexpr (std::is_same_v<Op, RSqrt>) return vdivq_f32(vdupq_n_f32(1.0F), vsqrtq_f32(v));
    if constexpr (std::is_same_v<Op, Floor>) return vrndmq_f32(v);
    if constexpr (std::is_same_v<Op, Ceil>) return vrndpq_f32(v);
    if constexpr (std::is_same_v<Op, Trunc>) return vrndq_f32(v);
    if constexpr (std::is_same_v<Op, Round>) return vrndaq_f32(v);
#endif
  } else if constexpr (std::is_same_v<T, double>) {
#if defined(__aarch64__)
    if constexpr (std::is_same_v<Op, Abs>) return vabsq_f64(v);
    if constexpr (std::is_same_v<Op, Sqrt>) return vsqrtq_f64(v);
    if constexpr (std::is_same_v<Op, RSqrt>) return vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(v));
    if constexpr (std::is_same_v<Op, Floor>) return vrndmq_f64(v);
    if constexpr (std::is_same_v<Op, Ceil>) return vrndpq_f64(v);
    if constexpr (std::is_same_v<Op, Trunc>) return vrndq_f64(v);
    if constexpr (std::is_same_v<Op, Round>) return vrndaq_f64(v);
#endif
  } else if constexpr (std::is_same_v<T, int> && std::is_same_v<Op, Abs>) {
    return vabsq_s32(v);
  }
}

}  // namespace md::detail
