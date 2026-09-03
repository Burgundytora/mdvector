#pragma once

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
using md::where;
using md::select;
using md::logical_and;
using md::logical_or;
using md::logical_xor;
using md::floor;
using md::ceil;
using md::trunc;

using md::sum;
using md::sum_axis;
using md::mean_axis;
using md::prod_axis;
using md::min_axis;
using md::max_axis;
using md::prod;
using md::max;
using md::min;
using md::mean;
using md::any;
using md::all;
using md::argmin;
using md::argmax;
using md::median;
using md::variance;
using md::standard_deviation;

// // 为防止与STL冲突 屏蔽
// using md::vector;
// using md::array;
// using md::span;
// using md::view;
// using md::inplace_vector;
