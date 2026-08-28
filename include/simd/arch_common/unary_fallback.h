#pragma once

namespace md::detail {

// Used when the selected ISA has no corresponding vector instruction.
template <class T, class Op>
inline typename simd<T>::type map_simd_lanes(typename simd<T>::type v) {
  alignas(simd<T>::alignment) T lanes[simd<T>::pack_size];
  simd<T>::store(lanes, v);
  for (std::size_t i = 0; i < simd<T>::pack_size; ++i) {
    lanes[i] = scalar_op_unary_impl<T, Op>(lanes[i]);
  }
  return simd<T>::load(lanes);
}

}  // namespace md::detail
