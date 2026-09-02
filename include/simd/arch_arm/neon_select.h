#pragma once

namespace md::detail {

template <class T>
inline constexpr bool has_native_simd_select_v = true;

template <class T>
inline typename simd<T>::type native_simd_select(typename simd<T>::const_ref_type condition,
                                                 typename simd<T>::const_ref_type true_value,
                                                 typename simd<T>::const_ref_type false_value) {
  if constexpr (std::is_same_v<T, float>) {
    const auto mask = vmvnq_u32(vceqq_f32(condition, vdupq_n_f32(0.0F)));
    return vbslq_f32(mask, true_value, false_value);
  } else if constexpr (std::is_same_v<T, double>) {
    const auto mask = vmvnq_u64(vceqq_f64(condition, vdupq_n_f64(0.0)));
    return vbslq_f64(mask, true_value, false_value);
  } else {
    const auto mask = vmvnq_u32(vceqq_s32(condition, vdupq_n_s32(0)));
    return vbslq_s32(mask, true_value, false_value);
  }
}

}  // namespace md::detail
