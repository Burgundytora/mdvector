#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "simd_arch_select.h"

namespace md {

struct Add;
struct Sub;
struct Mul;
struct Div;
struct Pow;
struct Hypot;
struct Fmod;

namespace detail {

template <class>
inline constexpr bool binary_dependent_false_v = false;

template <class T, class Cal>
inline T scalar_op_binary_impl(T l, T r) {
  if constexpr (std::is_same_v<Cal, Add>) {
    return l + r;
  } else if constexpr (std::is_same_v<Cal, Sub>) {
    return l - r;
  } else if constexpr (std::is_same_v<Cal, Mul>) {
    return l * r;
  } else if constexpr (std::is_same_v<Cal, Div>) {
    return l / r;
  } else if constexpr (std::is_same_v<Cal, Pow>) {
    return static_cast<T>(std::pow(l, r));
  } else if constexpr (std::is_same_v<Cal, Hypot>) {
    return static_cast<T>(std::hypot(l, r));
  } else if constexpr (std::is_same_v<Cal, Fmod>) {
    return static_cast<T>(std::fmod(l, r));
  } else {
    static_assert(binary_dependent_false_v<Cal>, "unsupported binary operation");
  }
}

// No portable SIMD instruction exists for these functions. Keep evaluation
// pack-based and use the standard library only for the individual lanes.
template <class T, class Cal>
inline typename simd<T>::type map_binary_simd_lanes(typename simd<T>::const_ref_type l,
                                                    typename simd<T>::const_ref_type r) {
  alignas(simd<T>::alignment) std::array<T, simd<T>::pack_size> lhs{};
  alignas(simd<T>::alignment) std::array<T, simd<T>::pack_size> rhs{};
  simd<T>::store(lhs.data(), l);
  simd<T>::store(rhs.data(), r);
  for (size_t i = 0; i < simd<T>::pack_size; ++i) {
    lhs[i] = scalar_op_binary_impl<T, Cal>(lhs[i], rhs[i]);
  }
  return simd<T>::load(lhs.data());
}

}  // namespace detail

template <class T, class Cal>
inline T scalar_op_binary(T l, T r) {
  return detail::scalar_op_binary_impl<T, Cal>(l, r);
}

template <class T, class Cal>
inline typename simd<T>::type simd_op_binary(typename simd<T>::const_ref_type l, typename simd<T>::const_ref_type r) {
  if constexpr (std::is_same_v<Cal, Add>) {
    return simd<T>::add(l, r);
  } else if constexpr (std::is_same_v<Cal, Sub>) {
    return simd<T>::sub(l, r);
  } else if constexpr (std::is_same_v<Cal, Mul>) {
    return simd<T>::mul(l, r);
  } else if constexpr (std::is_same_v<Cal, Div>) {
    return simd<T>::div(l, r);
  } else if constexpr (std::is_same_v<Cal, Pow> || std::is_same_v<Cal, Hypot> || std::is_same_v<Cal, Fmod>) {
    return detail::map_binary_simd_lanes<T, Cal>(l, r);
  } else {
    static_assert(detail::binary_dependent_false_v<Cal>, "unsupported binary operation");
  }
}

}  // namespace md
