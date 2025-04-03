#ifndef __TEST_SET_H__
#define __TEST_SET_H__

#include <iostream>
#include <vector>

using std::vector;

constexpr bool do_add = true;
constexpr bool do_sub = true;
constexpr bool do_mul = true;
constexpr bool do_div = false;

constexpr size_t points = 0.5E9;

struct TestPoint {
  TestPoint(size_t dim1, size_t dim2) : dim1_(dim1), dim2_(dim2) {
    total_element_ = dim1 * dim2;
    loop_ = points / total_element_;
    total_cal_ = points * (do_add + do_sub + do_mul + do_div);
  }

  size_t loop_;
  size_t dim1_;
  size_t dim2_;
  size_t total_element_;
  size_t total_cal_;
};

vector<TestPoint> all_test_points = {TestPoint(1, 10),  TestPoint(1, 50),   TestPoint(2, 60),   TestPoint(3, 70),
                                     TestPoint(5, 100), TestPoint(10, 100), TestPoint(100, 100)};

size_t loop;
size_t dim1;
size_t dim2;
size_t total_element;
size_t total_cal;
#endif  // __TEST_SET_H__