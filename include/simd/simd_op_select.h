#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "simd_arch_select.h"

namespace md {

template <class T>
inline typename simd<T>::type simd_op_select(typename simd<T>::const_ref_type condition,
                                             typename simd<T>::const_ref_type true_value,
                                             typename simd<T>::const_ref_type false_value) {
  if constexpr (detail::has_native_simd_select_v<T>) {
    return detail::native_simd_select<T>(condition, true_value, false_value);
  } else {
    alignas(simd<T>::alignment) std::array<T, simd<T>::pack_size> conditions{};
    alignas(simd<T>::alignment) std::array<T, simd<T>::pack_size> true_values{};
    alignas(simd<T>::alignment) std::array<T, simd<T>::pack_size> false_values{};
    simd<T>::store(conditions.data(), condition);
    simd<T>::store(true_values.data(), true_value);
    simd<T>::store(false_values.data(), false_value);
    for (size_t i = 0; i < simd<T>::pack_size; ++i) {
      true_values[i] = conditions[i] != T{} ? true_values[i] : false_values[i];
    }
    return simd<T>::load(true_values.data());
  }
}

}  // namespace md
