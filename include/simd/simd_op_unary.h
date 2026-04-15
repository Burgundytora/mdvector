#ifndef __MDVECTOR_OP_UNARY__
#define __MDVECTOR_OP_UNARY__

#include "simd_arch_select.h"
#include <cmath>

namespace md {

// ============================================================================
// 一元操作标签
// ============================================================================

// 基本数学函数
struct Neg;    // 取负: -x
struct Abs;    // 绝对值: |x|
struct Sqrt;   // 平方根: √x
struct Cbrt;   // 立方根: ∛x
struct RSqrt;  // 平方根倒数: 1/√x

// 指数和对数
struct Exp;    // e^x
struct Exp2;   // 2^x
struct Expm1;  // e^x - 1
struct Log;    // ln(x)
struct Log2;   // log2(x)
struct Log10;  // log10(x)
struct Log1p;  // ln(1+x)

// 三角函数
struct Sin;   // sin(x)
struct Cos;   // cos(x)
struct Tan;   // tan(x)
struct Asin;  // arcsin(x)
struct Acos;  // arccos(x)
struct Atan;  // arctan(x)

// 双曲函数
struct Sinh;   // sinh(x)
struct Cosh;   // cosh(x)
struct Tanh;   // tanh(x)
struct Asinh;  // arcsinh(x)
struct Acosh;  // arccosh(x)
struct Atanh;  // arctanh(x)

// 取整函数
struct Floor;  // ⌊x⌋
struct Ceil;   // ⌈x⌉
struct Trunc;  // 向零取整
struct Round;  // 四舍五入

// 其他
struct Erf;     // 误差函数
struct Erfc;    // 互补误差函数
struct Tgamma;  // Gamma 函数
struct Lgamma;  // Log Gamma 函数

// ============================================================================
// SIMD 一元操作分发
// ============================================================================
template <class T, class Op>
inline typename simd<T>::type simd_op_unary(typename simd<T>::const_ref_type v) {
  // 基本数学
  if constexpr (std::is_same_v<Op, Neg>) {
    return simd<T>::neg(v);
  } else if constexpr (std::is_same_v<Op, Abs>) {
    return simd<T>::abs(v);
  } else if constexpr (std::is_same_v<Op, Sqrt>) {
    return simd<T>::sqrt(v);
  } else if constexpr (std::is_same_v<Op, Cbrt>) {
    return simd<T>::cbrt(v);
  } else if constexpr (std::is_same_v<Op, RSqrt>) {
    return simd<T>::rsqrt(v);
  }
  // 指数和对数
  else if constexpr (std::is_same_v<Op, Exp>) {
    return simd<T>::exp(v);
  } else if constexpr (std::is_same_v<Op, Exp2>) {
    return simd<T>::exp2(v);
  } else if constexpr (std::is_same_v<Op, Expm1>) {
    return simd<T>::expm1(v);
  } else if constexpr (std::is_same_v<Op, Log>) {
    return simd<T>::log(v);
  } else if constexpr (std::is_same_v<Op, Log2>) {
    return simd<T>::log2(v);
  } else if constexpr (std::is_same_v<Op, Log10>) {
    return simd<T>::log10(v);
  } else if constexpr (std::is_same_v<Op, Log1p>) {
    return simd<T>::log1p(v);
  }
  // 三角函数
  else if constexpr (std::is_same_v<Op, Sin>) {
    return simd<T>::sin(v);
  } else if constexpr (std::is_same_v<Op, Cos>) {
    return simd<T>::cos(v);
  } else if constexpr (std::is_same_v<Op, Tan>) {
    return simd<T>::tan(v);
  } else if constexpr (std::is_same_v<Op, Asin>) {
    return simd<T>::asin(v);
  } else if constexpr (std::is_same_v<Op, Acos>) {
    return simd<T>::acos(v);
  } else if constexpr (std::is_same_v<Op, Atan>) {
    return simd<T>::atan(v);
  }
  // 双曲函数
  else if constexpr (std::is_same_v<Op, Sinh>) {
    return simd<T>::sinh(v);
  } else if constexpr (std::is_same_v<Op, Cosh>) {
    return simd<T>::cosh(v);
  } else if constexpr (std::is_same_v<Op, Tanh>) {
    return simd<T>::tanh(v);
  } else if constexpr (std::is_same_v<Op, Asinh>) {
    return simd<T>::asinh(v);
  } else if constexpr (std::is_same_v<Op, Acosh>) {
    return simd<T>::acosh(v);
  } else if constexpr (std::is_same_v<Op, Atanh>) {
    return simd<T>::atanh(v);
  }
  // 取整函数
  else if constexpr (std::is_same_v<Op, Floor>) {
    return simd<T>::floor(v);
  } else if constexpr (std::is_same_v<Op, Ceil>) {
    return simd<T>::ceil(v);
  } else if constexpr (std::is_same_v<Op, Trunc>) {
    return simd<T>::trunc(v);
  } else if constexpr (std::is_same_v<Op, Round>) {
    return simd<T>::round(v);
  }
  // 其他
  else if constexpr (std::is_same_v<Op, Erf>) {
    return simd<T>::erf(v);
  } else if constexpr (std::is_same_v<Op, Erfc>) {
    return simd<T>::erfc(v);
  } else if constexpr (std::is_same_v<Op, Tgamma>) {
    return simd<T>::tgamma(v);
  } else if constexpr (std::is_same_v<Op, Lgamma>) {
    return simd<T>::lgamma(v);
  } else {
    static_assert(false, "simd_op_unary: unsupported operation");
  }
}

// ============================================================================
// 标量回退（用于不支持 SIMD 的数学函数）
// ============================================================================
template <class T, class Op>
inline T scalar_op_unary(T v) {
  if constexpr (std::is_same_v<Op, Neg>) {
    return -v;
  } else if constexpr (std::is_same_v<Op, Abs>) {
    return std::abs(v);
  } else if constexpr (std::is_same_v<Op, Sqrt>) {
    return std::sqrt(v);
  } else if constexpr (std::is_same_v<Op, Cbrt>) {
    return std::cbrt(v);
  } else if constexpr (std::is_same_v<Op, Exp>) {
    return std::exp(v);
  } else if constexpr (std::is_same_v<Op, Log>) {
    return std::log(v);
  } else if constexpr (std::is_same_v<Op, Log10>) {
    return std::log10(v);
  } else if constexpr (std::is_same_v<Op, Sin>) {
    return std::sin(v);
  } else if constexpr (std::is_same_v<Op, Cos>) {
    return std::cos(v);
  } else if constexpr (std::is_same_v<Op, Tan>) {
    return std::tan(v);
  } else if constexpr (std::is_same_v<Op, Floor>) {
    return std::floor(v);
  } else if constexpr (std::is_same_v<Op, Ceil>) {
    return std::ceil(v);
  } else if constexpr (std::is_same_v<Op, Trunc>) {
    return std::trunc(v);
  } else if constexpr (std::is_same_v<Op, Round>) {
    return std::round(v);
  } else {
    static_assert(false, "scalar_op_unary: unsupported operation");
  }
}

}  // namespace md

#endif  // __MDVECTOR_OP_UNARY__