#ifndef __MDVECTOR_INCLUDE__
#define __MDVECTOR_INCLUDE__

#include "core/md_vector.h"
#include "core/md_array.h"
#include "core/md_span.h"
#include "core/md_view.h"
#include "math/math_function.h"

namespace md {

template <size_t Rank>
using shape = std::array<size_t, Rank>;

}  // namespace md

template <typename T>
using vector_1d = md::vector<T, 1>;

template <typename T>
using vector_2d = md::vector<T, 2>;

template <typename T>
using vector_3d = md::vector<T, 3>;

using md::all;
using md::slice;
using md::print_mdspan;
using md::print_simd_type;

using md::abs;
using md::acos;
using md::asin;
using md::atan;
using md::cos;
using md::cosh;
using md::ln;
using md::log10;
using md::sin;
using md::sinh;
using md::sqrt;
using md::cbrt;
using md::tan;
using md::tanh;
using md::exp;
using md::pow;
using md::fmod;
using md::hypot;
using md::floor;
using md::ceil;
using md::trunc;

using md::sum;
using md::prod;
using md::max;
using md::min;
using md::mean;
using md::median;
using md::variance;
using md::standard_deviation;

// // 为防止与STL冲突 屏蔽
// using md::vector;
// using md::array;
// using md::span;
// using md::view;
// using md::inplace_vector;

#endif  // __MDVECTOR_INCLUDE__