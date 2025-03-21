#ifndef __TEST_SET_H__
#define __TEST_SET_H__

#include <iostream>

#include "src/header/time_cost.h"

constexpr bool do_add = true;
constexpr bool do_sub = true;
constexpr bool do_mul = true;
constexpr bool do_div = true;

size_t loop = 1000000;
size_t dim1 = 10;
size_t dim2 = 100;
size_t total_element = dim1 * dim2;

size_t total_cal = loop * total_element * (do_add + do_sub + do_mul + do_div);

#endif  // __TEST_SET_H__