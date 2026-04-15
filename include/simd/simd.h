#ifndef __MDVECTOR_SIMD__
#define __MDVECTOR_SIMD__

#include "simd_arch_select.h"
#include "simd_io_policy.h"
#include "simd_op_binary.h"
#include "simd_op_unary.h"

namespace md {

// 对齐
template <typename T>
size_t get_aligned_size(size_t size) {
  if constexpr (Numeric<T>) {
    return (size % simd<T>::pack_size == 0) ? size : ((size / simd<T>::pack_size) + 1) * simd<T>::pack_size;
  } else {
    return size;
  }
}

}  // namespace md

#endif  // __MDVECTOR_SIMD__