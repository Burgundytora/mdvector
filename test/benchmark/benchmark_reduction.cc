#include "include_md_all.h"

#include <chrono>
#include <iostream>
#include <numeric>

namespace {

volatile double benchmark_sink = 0.0;

template <typename Function>
double elapsed_ms(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  const double value = function();
  const auto end = std::chrono::steady_clock::now();
  benchmark_sink = benchmark_sink + value;
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_result(const char* name, double milliseconds) {
  std::cout << name << ": " << milliseconds << " ms\n";
}

}  // namespace

int main() {
  constexpr size_t rows = 512;
  constexpr size_t columns = 1024;
  constexpr size_t repetitions = 100;
  md::vector<double, 2> values(rows, columns);
  values.fill(0.25);

  const double full_sum = elapsed_ms([&] {
    double checksum = 0.0;
    for (size_t repeat = 0; repeat < repetitions; ++repeat) checksum += md::sum(values);
    return checksum;
  });

  const double full_sum_scalar = elapsed_ms([&] {
    double checksum = 0.0;
    for (size_t repeat = 0; repeat < repetitions; ++repeat)
      checksum += std::accumulate(values.data(), values.data() + values.size(), 0.0);
    return checksum;
  });

  const double last_axis = elapsed_ms([&] {
    double checksum = 0.0;
    for (size_t repeat = 0; repeat < repetitions; ++repeat) {
      const auto result = md::sum_axis<1>(values);
      checksum += result.data()[repeat % rows];
    }
    return checksum;
  });

  const double last_axis_scalar = elapsed_ms([&] {
    double checksum = 0.0;
    for (size_t repeat = 0; repeat < repetitions; ++repeat) {
      md::vector<double, 1> result(rows);
      for (size_t row = 0; row < rows; ++row)
        result.data()[row] = std::accumulate(values.data() + row * columns,
                                             values.data() + (row + 1) * columns, 0.0);
      checksum += result.data()[repeat % rows];
    }
    return checksum;
  });

  const double first_axis = elapsed_ms([&] {
    double checksum = 0.0;
    for (size_t repeat = 0; repeat < repetitions; ++repeat) {
      const auto result = md::sum_axis<0>(values);
      checksum += result.data()[repeat % columns];
    }
    return checksum;
  });

  const double first_axis_scalar = elapsed_ms([&] {
    double checksum = 0.0;
    for (size_t repeat = 0; repeat < repetitions; ++repeat) {
      md::vector<double, 1> result(columns);
      std::fill_n(result.data(), columns, 0.0);
      for (size_t row = 0; row < rows; ++row)
        for (size_t column = 0; column < columns; ++column)
          result.data()[column] += values.data()[row * columns + column];
      checksum += result.data()[repeat % columns];
    }
    return checksum;
  });

  const double fused_expression = elapsed_ms([&] {
    double checksum = 0.0;
    for (size_t repeat = 0; repeat < repetitions; ++repeat) {
      const auto result = md::sum_axis<0>(values * 1.5 + 0.25);
      checksum += result.data()[repeat % columns];
    }
    return checksum;
  });

  print_result("sum", full_sum);
  print_result("sum scalar", full_sum_scalar);
  print_result("sum_axis<1>", last_axis);
  print_result("sum_axis<1> scalar", last_axis_scalar);
  print_result("sum_axis<0>", first_axis);
  print_result("sum_axis<0> scalar", first_axis_scalar);
  print_result("sum_axis<0> fused expression", fused_expression);
  std::cout << "sink: " << benchmark_sink << '\n';
  return 0;
}
