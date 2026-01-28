#ifndef __MDVECTOR_INCLUDE_MD_H__
#define __MDVECTOR_INCLUDE_MD_H__

#include "vector.h"
#include "inplace_vector.h"
#include "array.h"
#include "span.h"
#include "view.h"

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

#endif  // __MDVECTOR_INCLUDE_MD_H__