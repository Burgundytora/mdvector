#include <iostream>

#include "src/header/time_cost.h"

constexpr bool do_add = true;
constexpr bool do_sub = true;
constexpr bool do_mul = true;
constexpr bool do_div = true;

size_t loop = 1000000;
size_t dim1 = 3;
size_t dim2 = 59;
size_t total_element = dim1 * dim2;