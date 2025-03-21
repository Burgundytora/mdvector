#ifndef __TEST_SET_H__
#define __TEST_SET_H__

#include <iostream>
#include <vector>

using std::vector;

#include "src/header/time_cost.h"

constexpr bool do_add = true;
constexpr bool do_sub = true;
constexpr bool do_mul = true;
constexpr bool do_div = true;

struct TestPoint {
  TestPoint(size_t loop, size_t dim1, size_t dim2) : loop_(loop), dim1_(dim1), dim2_(dim2) {
    total_element_ = dim1 * dim2;
    total_cal_ = loop * total_element_ * (do_add + do_sub + do_mul + do_div);
  }

  size_t loop_;
  size_t dim1_;
  size_t dim2_;
  size_t total_element_;
  size_t total_cal_;
};

vector<TestPoint> all_test_points = {TestPoint(10000000, 1, 50), TestPoint(1000000, 3, 80), TestPoint(10000, 100, 100)};

size_t loop;
size_t dim1;
size_t dim2;
size_t total_element;
size_t total_cal;
#endif  // __TEST_SET_H__