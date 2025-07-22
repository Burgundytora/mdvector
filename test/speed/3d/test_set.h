#ifndef __TEST_SET_3D_H__
#define __TEST_SET_3D_H__

#include <array>
#include <iostream>
#include <vector>

using std::array;
using std::vector;

constexpr bool do_add = true;
constexpr bool do_sub = true;
constexpr bool do_mul = true;
constexpr bool do_div = true;

constexpr size_t points = 3E8;

struct TestPoint {
  TestPoint() = delete;

  constexpr TestPoint(size_t dim1, size_t dim2, size_t dim3)
      : dim1_(dim1),
        dim2_(dim2),
        dim3_(dim3),
        total_element_(dim1 * dim2 * dim3),
        loop_(points / total_element_),
        total_cal_(points * (do_add + do_sub + do_mul + do_div)) {}

  const size_t dim1_;
  const size_t dim2_;
  size_t dim3_;
  const size_t total_element_;
  const size_t loop_;
  const size_t total_cal_;
};

constexpr array<TestPoint, 7> all_test_points = {TestPoint(2, 2, 2),   TestPoint(3, 3, 3),    TestPoint(5, 5, 5),
                                                 TestPoint(7, 7, 7),   TestPoint(10, 10, 10), TestPoint(20, 20, 20),
                                                 TestPoint(30, 30, 30)};

size_t loop;
size_t dim1;
size_t dim2;
size_t dim3;
size_t total_element;
size_t total_cal;
#endif  // __TEST_SET_3D_H__