
#include "include_md_all.h"

#include <cmath>
#include <iostream>
#include <type_traits>

template <typename T>
bool all_equal(const md::vector<T, 2>& values, T expected, T tolerance = T{}) {
  for (const auto value : values) {
    if (std::abs(value - expected) > tolerance) {
      std::cerr << "expected " << expected << ", got " << value << '\n';
      return false;
    }
  }
  return true;
}

int main() {
  md::vector<double, 2> a(1, 3);
  md::vector<double, 2> b(1, 3);
  md::vector<double, 2> c(1, 3);
  md::vector<double, 2> d(1, 3);
  md::vector<double, 2> res(1, 3);
  a.fill(1.0);
  b.fill(2.0);
  c.fill(3.0);
  d.fill(4.0);

  auto tmp_plus = a + b;
  auto tmp_multi = a * b;

  // Containers are always referenced; lightweight expression nodes are owned.
  // These assertions prevent a future change from copying array storage or
  // reintroducing references to temporary expression nodes.
  static_assert(std::is_same_v<md::AutoType<decltype(a)>, const decltype(a)&>);
  static_assert(std::is_same_v<md::AutoType<decltype(tmp_plus)>, decltype(tmp_plus)>);
  using unary_node_type = md::unary_expr<md::Sqrt, decltype(tmp_plus), double>;
  static_assert(std::is_same_v<md::AutoType<unary_node_type>, unary_node_type>);

  res = c + tmp_plus;
  if (!all_equal(res, 6.0)) return 1;
  res = tmp_multi;
  if (!all_equal(res, 2.0)) return 1;

  // Both child nodes are temporaries. The resulting expression is deliberately
  // evaluated in a later statement to exercise their lifetime.
  auto expr = (a + b) * (c + d);
  res = expr;
  res.print();
  if (!all_equal(res, 21.0)) return 1;

  // Deep binary tree with scalar wrappers, also evaluated after construction.
  auto deep_expr = ((a + b) * (c + d) - (a * d + b / c)) / (a + 1.0);
  res = deep_expr;
  res.print();
  if (!all_equal(res, (21.0 - (4.0 + 2.0 / 3.0)) / 2.0, 1e-12)) return 1;

  // Leaf containers stay referenced, so a delayed expression observes later
  // data changes instead of holding an expensive container copy.
  auto live_expr = a + b;
  a.fill(5.0);
  res = live_expr;
  if (!all_equal(res, 7.0)) return 1;

  return 0;
}
