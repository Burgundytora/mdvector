#include <iostream>

#include "src/header/time_cost.h"

constexpr bool do_add = false;
constexpr bool do_sub = true;
constexpr bool do_mul = false;
constexpr bool do_div = false;

size_t loop = 10000000;
size_t dim1 = 3;
size_t dim2 = 59;
size_t total_element = dim1 * dim2;