#include "include_md_all.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

bool close(double lhs, double rhs) { return std::abs(lhs - rhs) < 1e-12; }

template <typename F>
bool throws_invalid_argument(F&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  md::vector<double, 2> values(2, 3);
  values.set_arange(1.0, 1.0);

  if (!close(md::sum(values), 21.0) || !close(md::prod(values), 720.0) ||
      !close(md::min(values), 1.0) || !close(md::max(values), 6.0) ||
      !close(md::mean(values), 3.5) || md::argmin(values) != 0 || md::argmax(values) != 5)
    return 1;

  const auto expression = values * 2.0 - 3.0;
  if (!close(md::sum(expression), 24.0) || !close(md::min(expression), -1.0) ||
      !close(md::max(expression), 9.0) || md::argmin(expression) != 0 || md::argmax(expression) != 5)
    return 2;
  const auto expression_min = md::min_axis<1>(expression);
  const auto expression_max = md::max_axis<1>(expression);
  const auto expression_argmin = md::argmin_axis<1>(expression);
  const auto expression_argmax = md::argmax_axis<1>(expression);
  if (expression_min[0] != -1.0 || expression_min[1] != 5.0 ||
      expression_max[0] != 3.0 || expression_max[1] != 9.0 ||
      expression_argmin[0] != 0 || expression_argmin[1] != 0 ||
      expression_argmax[0] != 2 || expression_argmax[1] != 2)
    return 18;

  const auto sum0 = md::sum_axis<0>(values);
  const auto sum1 = md::sum_axis<1>(values);
  const auto product0 = md::prod_axis<0>(values);
  const auto product1 = md::prod_axis<1>(values);
  const auto min0 = md::min_axis<0>(values);
  const auto max1 = md::max_axis<1>(values);
  const auto mean0 = md::mean_axis<0>(values);
  if (sum0[0] != 5.0 || sum0[1] != 7.0 || sum0[2] != 9.0 ||
      sum1[0] != 6.0 || sum1[1] != 15.0 || product0[0] != 4.0 ||
      product0[1] != 10.0 || product0[2] != 18.0 || product1[0] != 6.0 ||
      product1[1] != 120.0 || min0[0] != 1.0 || min0[2] != 3.0 ||
      max1[0] != 3.0 || max1[1] != 6.0 || !close(mean0[0], 2.5) || !close(mean0[2], 4.5))
    return 3;

  md::vector<int, 2> truth(2, 3);
  truth[0, 0] = 0; truth[0, 1] = 1; truth[0, 2] = 0;
  truth[1, 0] = 1; truth[1, 1] = 1; truth[1, 2] = 1;
  if (!md::any(truth) || md::all(truth)) return 4;
  const auto any0 = md::any_axis<0>(truth);
  const auto all0 = md::all_axis<0>(truth);
  const auto any1 = md::any_axis<1>(truth);
  const auto all1 = md::all_axis<1>(truth);
  if (any0[0] != 1 || any0[1] != 1 || any0[2] != 1 || all0[0] != 0 || all0[1] != 1 ||
      all0[2] != 0 || any1[0] != 1 || any1[1] != 1 || all1[0] != 0 || all1[1] != 1)
    return 5;
  const auto expression_any = md::any_axis<1>(truth - 1);
  const auto expression_all = md::all_axis<1>(truth + 1);
  if (expression_any[0] != 1 || expression_any[1] != 0 ||
      expression_all[0] != 1 || expression_all[1] != 1)
    return 19;

  const auto argmin0 = md::argmin_axis<0>(values);
  const auto argmax0 = md::argmax_axis<0>(values);
  const auto argmin1 = md::argmin_axis<1>(values);
  const auto argmax1 = md::argmax_axis<1>(values);
  static_assert(std::is_same_v<typename decltype(argmin0)::value_type, size_t>);
  static_assert(std::is_same_v<typename decltype(any0)::value_type, int>);
  if (argmin0[0] != 0 || argmin0[2] != 0 || argmax0[0] != 1 || argmax0[2] != 1 ||
      argmin1[0] != 0 || argmin1[1] != 0 || argmax1[0] != 2 || argmax1[1] != 2)
    return 6;

  auto strided = values.view(all(), slice(0, 2, 2));
  if (md::sum(strided) != 14.0 || md::sum_axis<1>(strided)[0] != 4.0 ||
      md::sum_axis<1>(strided)[1] != 10.0)
    return 7;
  const auto& const_values = values;
  const auto read_only = const_values.view(all(), all());
  if (md::sum(read_only) != 21.0 || md::max_axis<0>(read_only)[2] != 6.0) return 8;

  auto reversed = values.view(slice(1, -1, 0), all());
  if (reversed.get_1d_index(1, 2) != -1 || md::sum(reversed) != 21.0 ||
      md::sum_axis<0>(reversed)[0] != 5.0 || md::sum_axis<1>(reversed)[0] != 15.0)
    return 12;

  md::vector<double, 2> unaligned_rows(2, 5);
  unaligned_rows.set_arange(1.0, 1.0);
  const auto unaligned_sum = md::sum_axis<1>(unaligned_rows);
  if (unaligned_sum[0] != 15.0 || unaligned_sum[1] != 40.0) return 13;

  md::vector<double, 2, std::layout_left> column_major(2, 3);
  for (size_t row = 0; row < 2; ++row)
    for (size_t column = 0; column < 3; ++column)
      column_major(row, column) = values(row, column);
  const auto column_sum0 = md::sum_axis<0>(column_major);
  const auto column_sum1 = md::sum_axis<1>(column_major * 2.0);
  if (column_sum0[0] != 5.0 || column_sum0[1] != 7.0 || column_sum0[2] != 9.0 ||
      column_sum1[0] != 12.0 || column_sum1[1] != 30.0 ||
      md::argmax_axis<0>(column_major)[2] != 1)
    return 14;

  md::vector<double, 2> empty(2, 0);
  if (md::sum(empty) != 0.0 || md::prod(empty) != 1.0 || md::any(empty) || !md::all(empty)) return 9;
  const auto empty_sum = md::sum_axis<1>(empty);
  const auto empty_product = md::prod_axis<1>(empty);
  const auto empty_any = md::any_axis<1>(empty);
  const auto empty_all = md::all_axis<1>(empty);
  if (empty_sum[0] != 0.0 || empty_product[0] != 1.0 || empty_any[0] != 0 || empty_all[0] != 1)
    return 10;
  if (!throws_invalid_argument([&] { (void)md::min(empty); }) ||
      !throws_invalid_argument([&] { (void)md::max(empty); }) ||
      !throws_invalid_argument([&] { (void)md::mean(empty); }) ||
      !throws_invalid_argument([&] { (void)md::argmin(empty); }) ||
      !throws_invalid_argument([&] { (void)md::argmax(empty); }) ||
      !throws_invalid_argument([&] { (void)md::min_axis<1>(empty); }) ||
      !throws_invalid_argument([&] { (void)md::max_axis<1>(empty); }) ||
      !throws_invalid_argument([&] { (void)md::mean_axis<1>(empty); }) ||
      !throws_invalid_argument([&] { (void)md::argmin_axis<1>(empty); }) ||
      !throws_invalid_argument([&] { (void)md::argmax_axis<1>(empty); }))
    return 11;

  md::vector<double, 3> empty_non_axis(2, 0, 3);
  if (md::sum_axis<0>(empty_non_axis).size() != 0) return 15;
  const auto empty_middle_sum = md::sum_axis<1>(empty_non_axis);
  if (empty_middle_sum.extents() != std::array<size_t, 2>{2, 3} ||
      md::sum(empty_middle_sum) != 0.0)
    return 16;

  md::vector<double, 3> cube(2, 3, 4);
  cube.set_arange(1.0, 1.0);
  const auto cube_sum0 = md::sum_axis<0>(cube);
  const auto cube_sum1 = md::sum_axis<1>(cube);
  const auto cube_sum2 = md::sum_axis<2>(cube);
  if (cube_sum0.extents() != std::array<size_t, 2>{3, 4} || cube_sum0(2, 3) != 36.0 ||
      cube_sum1.extents() != std::array<size_t, 2>{2, 4} || cube_sum1(1, 3) != 60.0 ||
      cube_sum2.extents() != std::array<size_t, 2>{2, 3} || cube_sum2(1, 2) != 90.0 ||
      md::max_axis<1>(cube)(1, 3) != 24.0 || md::argmax_axis<1>(cube)(1, 3) != 2)
    return 20;

  const std::vector<int> standard_values{3, 0, -2, 4};
  if (md::sum(standard_values) != 5 || md::prod(standard_values) != 0 ||
      md::min(standard_values) != -2 || md::max(standard_values) != 4 ||
      md::mean(standard_values) != 1 || !md::any(standard_values) || md::all(standard_values) ||
      md::argmin(standard_values) != 2 || md::argmax(standard_values) != 3)
    return 17;

  return 0;
}
