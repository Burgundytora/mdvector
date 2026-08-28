#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace md {

struct Neg;
struct Abs;
struct Sqrt;
struct Cbrt;
struct RSqrt;
struct Exp;
struct Exp2;
struct Expm1;
struct Log;
struct Log2;
struct Log10;
struct Log1p;
struct Sin;
struct Cos;
struct Tan;
struct Asin;
struct Acos;
struct Atan;
struct Sinh;
struct Cosh;
struct Tanh;
struct Asinh;
struct Acosh;
struct Atanh;
struct Floor;
struct Ceil;
struct Trunc;
struct Round;
struct Erf;
struct Erfc;
struct Tgamma;
struct Lgamma;

namespace detail {

template <class>
inline constexpr bool dependent_false_v = false;

template <class T, class Op>
inline T scalar_op_unary_impl(T v) {
  if constexpr (std::is_same_v<Op, Neg>) {
    return -v;
  } else if constexpr (std::is_same_v<Op, Abs>) {
    return static_cast<T>(std::abs(v));
  } else if constexpr (std::is_same_v<Op, Sqrt>) {
    return static_cast<T>(std::sqrt(v));
  } else if constexpr (std::is_same_v<Op, Cbrt>) {
    return static_cast<T>(std::cbrt(v));
  } else if constexpr (std::is_same_v<Op, RSqrt>) {
    return static_cast<T>(T{1} / std::sqrt(v));
  } else if constexpr (std::is_same_v<Op, Exp>) {
    return static_cast<T>(std::exp(v));
  } else if constexpr (std::is_same_v<Op, Exp2>) {
    return static_cast<T>(std::exp2(v));
  } else if constexpr (std::is_same_v<Op, Expm1>) {
    return static_cast<T>(std::expm1(v));
  } else if constexpr (std::is_same_v<Op, Log>) {
    return static_cast<T>(std::log(v));
  } else if constexpr (std::is_same_v<Op, Log2>) {
    return static_cast<T>(std::log2(v));
  } else if constexpr (std::is_same_v<Op, Log10>) {
    return static_cast<T>(std::log10(v));
  } else if constexpr (std::is_same_v<Op, Log1p>) {
    return static_cast<T>(std::log1p(v));
  } else if constexpr (std::is_same_v<Op, Sin>) {
    return static_cast<T>(std::sin(v));
  } else if constexpr (std::is_same_v<Op, Cos>) {
    return static_cast<T>(std::cos(v));
  } else if constexpr (std::is_same_v<Op, Tan>) {
    return static_cast<T>(std::tan(v));
  } else if constexpr (std::is_same_v<Op, Asin>) {
    return static_cast<T>(std::asin(v));
  } else if constexpr (std::is_same_v<Op, Acos>) {
    return static_cast<T>(std::acos(v));
  } else if constexpr (std::is_same_v<Op, Atan>) {
    return static_cast<T>(std::atan(v));
  } else if constexpr (std::is_same_v<Op, Sinh>) {
    return static_cast<T>(std::sinh(v));
  } else if constexpr (std::is_same_v<Op, Cosh>) {
    return static_cast<T>(std::cosh(v));
  } else if constexpr (std::is_same_v<Op, Tanh>) {
    return static_cast<T>(std::tanh(v));
  } else if constexpr (std::is_same_v<Op, Asinh>) {
    return static_cast<T>(std::asinh(v));
  } else if constexpr (std::is_same_v<Op, Acosh>) {
    return static_cast<T>(std::acosh(v));
  } else if constexpr (std::is_same_v<Op, Atanh>) {
    return static_cast<T>(std::atanh(v));
  } else if constexpr (std::is_same_v<Op, Floor>) {
    return static_cast<T>(std::floor(v));
  } else if constexpr (std::is_same_v<Op, Ceil>) {
    return static_cast<T>(std::ceil(v));
  } else if constexpr (std::is_same_v<Op, Trunc>) {
    return static_cast<T>(std::trunc(v));
  } else if constexpr (std::is_same_v<Op, Round>) {
    return static_cast<T>(std::round(v));
  } else if constexpr (std::is_same_v<Op, Erf>) {
    return static_cast<T>(std::erf(v));
  } else if constexpr (std::is_same_v<Op, Erfc>) {
    return static_cast<T>(std::erfc(v));
  } else if constexpr (std::is_same_v<Op, Tgamma>) {
    return static_cast<T>(std::tgamma(v));
  } else if constexpr (std::is_same_v<Op, Lgamma>) {
    return static_cast<T>(std::lgamma(v));
  } else {
    static_assert(dependent_false_v<Op>, "unsupported unary operation");
  }
}

}  // namespace detail
}  // namespace md

#include "simd_arch_select.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(_M_AMD64)
#if defined(__AVX512F__)
#include "arch_x86/avx512_unary.h"
#elif defined(__AVX2__)
#include "arch_x86/avx2_unary.h"
#elif defined(__SSE4_1__)
#include "arch_x86/sse_unary.h"
#else
#include "arch_common/none_unary.h"
#endif
#elif defined(__arm__) || defined(__aarch64__)
#include "arch_arm/neon_unary.h"
#elif defined(__riscv)
#include "arch_risc/risc_v_unary.h"
#else
#include "arch_common/none_unary.h"
#endif

#include "arch_common/unary_fallback.h"

namespace md {
template <class T, class Op>
inline T scalar_op_unary(T v) {
  return detail::scalar_op_unary_impl<T, Op>(v);
}

template <class T, class Op>
inline typename simd<T>::type simd_op_unary(typename simd<T>::type v) {
  if constexpr (detail::has_native_simd_op_unary_v<T, Op>) {
    return detail::native_simd_op_unary<T, Op>(v);
  } else {
    return detail::map_simd_lanes<T, Op>(v);
  }
}

}  // namespace md
