#pragma once

#include <array>
#include <iostream>
#include <vector>

using std::array;
using std::vector;

constexpr bool do_add = true;
constexpr bool do_sub = true;
constexpr bool do_mul = true;
constexpr bool do_div = true;

constexpr size_t points = 2E8;

struct TestPoint {
  TestPoint() = delete;

  constexpr TestPoint(size_t dim1, size_t dim2)
      : dim1_(dim1),
        dim2_(dim2),
        total_element_(dim1 * dim2),
        loop_(points / total_element_),
        total_cal_(points * (do_add + do_sub + do_mul + do_div)) {}

  const size_t dim1_;
  const size_t dim2_;
  const size_t total_element_;
  const size_t loop_;
  const size_t total_cal_;
};

constexpr array<TestPoint, 10> all_test_points = {
    TestPoint(1, 4),    TestPoint(1, 10),    TestPoint(1, 50),    TestPoint(3, 70),      TestPoint(5, 100),
    TestPoint(10, 100), TestPoint(100, 100), TestPoint(500, 500), TestPoint(3000, 3000), TestPoint(5000, 5000)};

size_t loop;
size_t dim1;
size_t dim2;
size_t total_element;
size_t total_cal;
