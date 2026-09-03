#include "include_md_all.h"

#include <cmath>
#include <type_traits>

template <typename T>
bool close(T actual, T expected, T tolerance = static_cast<T>(1e-6)) {
  return std::abs(actual - expected) <= tolerance * std::max<T>(T{1}, std::abs(expected));
}

int main() {
  md::vector<double, 2> source(2, 6);
  source.set_arange(1.0, 1.0);

  // Temporary span/view descriptors must be owned by the expression tree.
  auto span_expr = md::sin(source.span(0, all()) + 1.0);
  md::vector<double, 1> span_result = span_expr;
  for (size_t i = 0; i < span_result.size(); ++i) {
    if (!close(span_result[i], std::sin(static_cast<double>(i + 2)), 1e-12)) return 1;
  }

  auto view_expr = source.view(all(), slice(0, 2, -1)) * 3.0;
  md::vector<double, 2> view_result = view_expr;
  for (size_t row = 0; row < 2; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      if (view_result(row, col) != source(row, col * 2) * 3.0) return 1;
    }
  }

  // Partially overlapping assignments require a temporary buffer.
  md::vector<double, 1> overlap(10);
  overlap.set_arange(1.0, 1.0);
  auto input = overlap.span(slice(0, 7));
  auto output = overlap.span(slice(1, 8));
  output = input + 100.0;
  if (overlap[0] != 1.0 || overlap[9] != 10.0) return 1;
  for (size_t i = 0; i < 8; ++i) {
    if (overlap[i + 1] != static_cast<double>(i + 101)) return 1;
  }

  // Exact in-place mappings remain valid and do not require an intermediate array.
  overlap = overlap + 1.0;
  if (overlap[0] != 2.0 || overlap[1] != 102.0) return 1;

  // An owning destination can still overlap a differently mapped view into
  // its storage. The owning/owning fast path must not hide this case.
  md::vector<double, 2> alias_owner(4, 4);
  alias_owner.set_arange(1.0, 1.0);
  md::view<double, 2> transposed(alias_owner.data(), {4, 4}, {1, 4});
  alias_owner = transposed + 0.0;
  for (size_t row = 0; row < 4; ++row) {
    for (size_t col = 0; col < 4; ++col) {
      if (alias_owner(row, col) != static_cast<double>(col * 4 + row + 1)) return 1;
    }
  }

  // Mixed element types promote to common_type and remain lazy/SIMD-evaluable.
  const size_t mixed_size = md::simd<double>::pack_size * 2 + 1;
  md::vector<float, 1> floats(mixed_size);
  md::vector<double, 1> doubles(mixed_size);
  md::vector<int, 1> integers(mixed_size);
  for (size_t i = 0; i < mixed_size; ++i) {
    floats[i] = static_cast<float>(i) + 0.25F;
    doubles[i] = static_cast<double>(i) * 0.5;
    integers[i] = static_cast<int>(i) - 2;
  }

  auto mixed_expr = floats + doubles * 2.0;
  static_assert(std::is_same_v<typename decltype(mixed_expr)::value_type, double>);
  md::vector<double, 1> mixed_result = mixed_expr;
  for (size_t i = 0; i < mixed_size; ++i) {
    if (!close(mixed_result[i], static_cast<double>(floats[i]) + doubles[i] * 2.0, 1e-12)) return 1;
  }

  auto promoted_scalar = integers + 0.5;
  static_assert(std::is_same_v<typename decltype(promoted_scalar)::value_type, double>);
  md::vector<float, 1> converted_result = promoted_scalar;
  for (size_t i = 0; i < mixed_size; ++i) {
    if (!close(converted_result[i], static_cast<float>(integers[i] + 0.5), 1e-6F)) return 1;
  }

  auto mixed_where = md::where(integers > 0, floats, doubles + 10.0);
  static_assert(std::is_same_v<typename decltype(mixed_where)::value_type, double>);
  md::vector<double, 1> where_result = mixed_where;
  for (size_t i = 0; i < mixed_size; ++i) {
    const double expected = integers[i] > 0 ? static_cast<double>(floats[i]) : doubles[i] + 10.0;
    if (!close(where_result[i], expected, 1e-12)) return 1;
  }

  md::vector<double, 2> reduction_input(2, 3);
  reduction_input.set_arange(1.0, 1.0);
  if (md::sum(reduction_input) != 21.0 || md::prod(reduction_input) != 720.0 ||
      md::min(reduction_input) != 1.0 || md::max(reduction_input) != 6.0 ||
      md::argmin(reduction_input) != 0 || md::argmax(reduction_input) != 5)
    return 1;
  auto row_sums = md::sum_axis<1>(reduction_input);
  auto col_products = md::prod_axis<0>(reduction_input);
  if (row_sums.extents() != std::array<size_t, 1>{2} || row_sums[0] != 6.0 || row_sums[1] != 15.0 ||
      col_products[0] != 4.0 || col_products[2] != 18.0)
    return 1;
  md::vector<int, 1> predicates(4);
  predicates[0] = 0; predicates[1] = 1; predicates[2] = 0; predicates[3] = 2;
  if (!md::any(predicates) || md::all(predicates)) return 1;
  predicates.fill(1);
  if (!md::all(predicates)) return 1;

  md::vector<double, 2> shape_a(2, 3);
  md::vector<double, 2> shape_b(3, 2);
#ifndef NDEBUG
  bool shape_error = false;
  try {
    auto invalid = shape_a + shape_b;
    (void)invalid;
  } catch (const std::invalid_argument&) {
    shape_error = true;
  }
  if (!shape_error) return 1;

  md::vector<double, 1> shape_source(6);
  md::vector<double, 1> shape_destination(5);
  bool destination_shape_error = false;
  try {
    auto destination = shape_destination.span(all());
    destination = shape_source + 1.0;
  } catch (const std::invalid_argument&) {
    destination_shape_error = true;
  }
  if (!destination_shape_error) return 1;
#else
  // Release deliberately skips shape checks; do not evaluate this expression.
  auto unchecked = shape_a + shape_b;
  (void)unchecked;
#endif

  return 0;
}
