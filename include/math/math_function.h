#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../expression/binary_expr.h"
#include "../expression/unary_expr.h"

namespace md {

// Element-wise mathematical functions. Unary functions are declared in
// unary_expr.h; these overloads cover the binary functions without creating
// an intermediate md::vector.
template <typename Derived, typename T, typename Y>
  requires std::convertible_to<Y, T>
inline auto exp(const base_expr<Derived, T>& exponent, Y base) {
  return binary_expr<T, T, Derived, Pow>(static_cast<T>(base), exponent.derived());
}

template <typename Derived, typename T, typename Y>
  requires std::convertible_to<Y, T>
inline auto pow(const base_expr<Derived, T>& value, Y exponent) {
  return binary_expr<T, Derived, T, Pow>(value.derived(), static_cast<T>(exponent));
}

template <typename L, typename R, typename T>
inline auto pow(const base_expr<L, T>& value, const base_expr<R, T>& exponent) {
  return binary_expr<T, L, R, Pow>(value.derived(), exponent.derived());
}

template <typename Derived, typename T, typename Y>
  requires std::convertible_to<Y, T>
inline auto fmod(const base_expr<Derived, T>& value, Y divisor) {
  return binary_expr<T, Derived, T, Fmod>(value.derived(), static_cast<T>(divisor));
}

template <typename L, typename R, typename T>
inline auto fmod(const base_expr<L, T>& value, const base_expr<R, T>& divisor) {
  return binary_expr<T, L, R, Fmod>(value.derived(), divisor.derived());
}

template <typename L, typename R, typename T>
inline auto hypot(const base_expr<L, T>& x, const base_expr<R, T>& y) {
  return binary_expr<T, L, R, Hypot>(x.derived(), y.derived());
}

template <typename X, typename Y, typename Z, typename T>
inline auto hypot(const base_expr<X, T>& x, const base_expr<Y, T>& y, const base_expr<Z, T>& z) {
  return hypot(hypot(x, y), z);
}

namespace detail {

template <typename Expr>
using reduction_value_t = std::remove_cv_t<typename std::remove_cvref_t<Expr>::value_type>;

// Logical linear access without base_expr::scalar_at().  scalar_at() loads a
// complete SIMD pack for every scalar request, which made strided and axis
// reductions unnecessarily expensive.
template <typename Expr>
inline reduction_value_t<Expr> reduction_value(const Expr& expr, size_t linear) {
  using expr_type = std::remove_cvref_t<Expr>;
  using value_type = reduction_value_t<Expr>;
  if constexpr (is_expression_node_v<expr_type>) {
    return static_cast<value_type>(expr[linear]);
  } else if constexpr (!std::is_same_v<typename expr_type::layout_type, std::layout_stride> &&
                       requires { expr.data(); }) {
    return static_cast<value_type>(expr.data()[linear]);
  } else {
    const auto indices = expr.get_md_index(linear);
    return std::apply(
        [&expr](auto... index) { return static_cast<value_type>(expr(index...)); }, indices);
  }
}

template <typename T>
inline constexpr bool simd_reducible_v = is_simd_supported_v<T>;

template <typename Expr>
inline constexpr bool reduction_pack_access_v =
    is_expression_node_v<std::remove_cvref_t<Expr>> ||
    !std::is_same_v<typename std::remove_cvref_t<Expr>::layout_type, std::layout_stride>;

template <bool Product, typename Expr>
inline reduction_value_t<Expr> arithmetic_range_reduce(const Expr& expr, size_t begin, size_t count) {
  using value_type = reduction_value_t<Expr>;
  value_type result = Product ? value_type{1} : value_type{};
  size_t offset = 0;
  if constexpr (simd_reducible_v<value_type> && reduction_pack_access_v<Expr>) {
    constexpr size_t pack_size = simd<value_type>::pack_size;
    // Container SIMD loads are aligned. Consume the short unaligned prefix
    // instead of disabling SIMD for the entire range (important for rows whose
    // length is not a multiple of the pack size).
    for (; offset < count && (begin + offset) % pack_size != 0; ++offset) {
      const value_type value = reduction_value(expr, begin + offset);
      if constexpr (Product) result *= value;
      else result += value;
    }
    if (offset + pack_size <= count) {
      auto accumulator = simd<value_type>::set1(Product ? value_type{1} : value_type{});
      for (; offset + pack_size <= count; offset += pack_size) {
        const auto values = expr.template load_simd<value_type>(begin + offset);
        if constexpr (Product)
          accumulator = simd<value_type>::mul(accumulator, values);
        else
          accumulator = simd<value_type>::add(accumulator, values);
      }
      alignas(simd<value_type>::alignment) value_type lanes[pack_size];
      simd<value_type>::store(lanes, accumulator);
      for (value_type lane : lanes) {
        if constexpr (Product) result *= lane;
        else result += lane;
      }
    }
  }
  for (; offset < count; ++offset) {
    const value_type value = reduction_value(expr, begin + offset);
    if constexpr (Product) result *= value;
    else result += value;
  }
  return result;
}

template <typename Expr, typename Visitor>
inline void visit_reduction_range(const Expr& expr, size_t begin, size_t count,
                                  Visitor&& visitor) {
  using value_type = reduction_value_t<Expr>;
  size_t offset = 0;
  if constexpr (simd_reducible_v<value_type> && reduction_pack_access_v<Expr>) {
    constexpr size_t pack_size = simd<value_type>::pack_size;
    alignas(simd<value_type>::alignment) value_type lanes[pack_size];
    for (; offset < count && (begin + offset) % pack_size != 0; ++offset)
      if (!visitor(reduction_value(expr, begin + offset), begin + offset)) return;
    for (; offset + pack_size <= count; offset += pack_size) {
      simd<value_type>::store(lanes, expr.template load_simd<value_type>(begin + offset));
      for (size_t lane = 0; lane < pack_size; ++lane)
        if (!visitor(lanes[lane], begin + offset + lane)) return;
    }
  }
  for (; offset < count; ++offset)
    if (!visitor(reduction_value(expr, begin + offset), begin + offset)) return;
}

template <typename Expr, typename Visitor>
inline void visit_reduction_values(const Expr& expr, Visitor&& visitor) {
  visit_reduction_range(expr, 0, expr.size(), std::forward<Visitor>(visitor));
}

template <bool Minimum, typename Expr>
inline reduction_value_t<Expr> extreme_range_reduce(const Expr& expr, size_t begin,
                                                    size_t count) {
  using value_type = reduction_value_t<Expr>;
  value_type result = reduction_value(expr, begin);
  visit_reduction_range(expr, begin + 1, count - 1,
                        [&result](value_type value, size_t) {
                          if constexpr (Minimum)
                            result = std::min(result, value);
                          else
                            result = std::max(result, value);
                          return true;
                        });
  return result;
}

template <bool Any, typename Expr>
inline int logical_range_reduce(const Expr& expr, size_t begin, size_t count) {
  int result = Any ? 0 : 1;
  visit_reduction_range(expr, begin, count, [&result](auto value, size_t) {
    if constexpr (Any) {
      if (static_cast<bool>(value)) {
        result = 1;
        return false;
      }
    } else if (!static_cast<bool>(value)) {
      result = 0;
      return false;
    }
    return true;
  });
  return result;
}

template <bool Minimum, typename Expr>
inline size_t arg_range_reduce(const Expr& expr, size_t begin, size_t count) {
  using value_type = reduction_value_t<Expr>;
  value_type best = reduction_value(expr, begin);
  size_t result = 0;
  visit_reduction_range(expr, begin + 1, count - 1,
                        [begin, &best, &result](value_type value, size_t index) {
                          const bool better = Minimum ? value < best : best < value;
                          if (better) {
                            best = value;
                            result = index - begin;
                          }
                          return true;
                        });
  return result;
}

template <typename Expr>
auto reduction_strides(const Expr& expr) {
  constexpr size_t rank = std::remove_cvref_t<Expr>::rank_;
  const auto shape = expr.extents();
  std::array<size_t, rank> stride{};
  if constexpr (std::is_same_v<typename std::remove_cvref_t<Expr>::layout_type,
                               std::layout_left>) {
    stride[0] = 1;
    for (size_t d = 1; d < rank; ++d) stride[d] = stride[d - 1] * shape[d - 1];
  } else {
    stride[rank - 1] = 1;
    for (size_t d = rank - 1; d-- > 0;) stride[d] = stride[d + 1] * shape[d + 1];
  }
  return stride;
}

template <size_t Axis, typename Result, typename Expr, typename ReduceRange>
auto axis_reduce(const Expr& expr, ReduceRange&& reduce_range) {
  constexpr size_t rank = Expr::rank_;
  constexpr size_t out_rank = rank - 1;
  const auto shape = expr.extents();
  std::array<size_t, out_rank> out_shape{};
  const auto logical_stride = reduction_strides(expr);
  for (size_t d = 0, out_d = 0; d < rank; ++d) if (d != Axis) out_shape[out_d++] = shape[d];

  md::vector<Result, out_rank> result(out_shape);
  for (size_t out = 0; out < result.size(); ++out) {
    size_t remaining = out;
    size_t base = 0;
    for (size_t d = out_rank; d-- > 0;) {
      const size_t source_d = d >= Axis ? d + 1 : d;
      const size_t index = remaining % shape[source_d];
      remaining /= shape[source_d];
      base += index * logical_stride[source_d];
    }
    result.data()[out] = static_cast<Result>(
        reduce_range(base, logical_stride[Axis], shape[Axis]));
  }
  return result;
}

// For a row-major source, reducing a non-contiguous axis output-by-output
// repeatedly jumps through memory. Traverse the source in physical order and
// update a contiguous output segment instead. Expression packs are evaluated
// only once and the output uses unaligned SIMD loads/stores where necessary.
template <size_t Axis, bool Product, typename Expr>
auto arithmetic_axis_reduce(const Expr& expr) {
  using value_type = reduction_value_t<Expr>;
  constexpr size_t rank = std::remove_cvref_t<Expr>::rank_;
  constexpr size_t out_rank = rank - 1;
  const auto shape = expr.extents();
  const auto source_stride = reduction_strides(expr);

  if (source_stride[Axis] == 1) {
    return axis_reduce<Axis, value_type>(
        expr, [&expr](size_t base, size_t, size_t count) {
          return arithmetic_range_reduce<Product>(expr, base, count);
        });
  }

  if constexpr (!std::is_same_v<typename std::remove_cvref_t<Expr>::layout_type,
                                std::layout_left>) {
    std::array<size_t, out_rank> out_shape{};
    for (size_t d = 0, out_d = 0; d < rank; ++d)
      if (d != Axis) out_shape[out_d++] = shape[d];

    md::vector<value_type, out_rank> result(out_shape);
    const value_type identity = Product ? value_type{1} : value_type{};
    std::fill_n(result.data(), result.size(), identity);
    if (result.size() == 0) return result;

    const size_t inner = source_stride[Axis];
    const size_t axis_count = shape[Axis];
    const size_t outer = result.size() / inner;
    for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
      value_type* output = result.data() + outer_index * inner;
      for (size_t axis_index = 0; axis_index < axis_count; ++axis_index) {
        const size_t source_begin = (outer_index * axis_count + axis_index) * inner;
        size_t i = 0;
        if constexpr (simd_reducible_v<value_type> && reduction_pack_access_v<Expr>) {
          constexpr size_t pack_size = simd<value_type>::pack_size;
          for (; i < inner && (source_begin + i) % pack_size != 0; ++i) {
            const value_type value = reduction_value(expr, source_begin + i);
            if constexpr (Product) output[i] *= value;
            else output[i] += value;
          }
          for (; i + pack_size <= inner; i += pack_size) {
            const auto values = expr.template load_simd<value_type>(source_begin + i);
            const auto previous = simd<value_type>::loadu(output + i);
            if constexpr (Product)
              simd<value_type>::storeu(output + i, simd<value_type>::mul(previous, values));
            else
              simd<value_type>::storeu(output + i, simd<value_type>::add(previous, values));
          }
        }
        for (; i < inner; ++i) {
          const value_type value = reduction_value(expr, source_begin + i);
          if constexpr (Product) output[i] *= value;
          else output[i] += value;
        }
      }
    }
    return result;
  } else {
    return axis_reduce<Axis, value_type>(
        expr, [&expr](size_t base, size_t stride, size_t count) {
          value_type result = Product ? value_type{1} : value_type{};
          for (size_t i = 0; i < count; ++i) {
            const value_type value = reduction_value(expr, base + i * stride);
            if constexpr (Product) result *= value;
            else result += value;
          }
          return result;
        });
  }
}

template <typename Container>
concept md_expression = requires {
  typename std::remove_cvref_t<Container>::value_type;
} && std::derived_from<std::remove_cvref_t<Container>,
                       base_expr<std::remove_cvref_t<Container>,
                                 typename std::remove_cvref_t<Container>::value_type>>;

}  // namespace detail

// Generic container reductions are retained for standard iterator-based
// containers. Expression-specific overloads below consume lazy trees directly.
template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
auto sum(const Container& c) {
  return std::reduce(c.begin(), c.end());
}

template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
auto prod(const Container& c) {
  return std::reduce(c.begin(), c.end(), typename Container::value_type(1), std::multiplies<>());
}

template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
auto max(const Container& c) {
  if (c.size() == 0) throw std::invalid_argument("max of empty container");
  return *std::max_element(c.begin(), c.end());
}

template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
auto min(const Container& c) {
  if (c.size() == 0) throw std::invalid_argument("min of empty container");
  return *std::min_element(c.begin(), c.end());
}

template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
auto mean(const Container& c) {
  if (c.size() == 0) throw std::invalid_argument("mean of empty container");
  return sum(c) / static_cast<typename Container::value_type>(c.size());
}

template <StatisticContainer Container>
auto variance(const Container& c) {
  if (c.size() < 2) throw std::invalid_argument("variance requires at least two elements");
  auto m = mean(c);
  long double sum_sq = std::accumulate(c.begin(), c.end(), 0.0L, [m](long double acc, auto val) {
    const long double diff = static_cast<long double>(val) - static_cast<long double>(m);
    return acc + diff * diff;
  });
  return static_cast<typename Container::value_type>(sum_sq / (c.size() - 1));
}

template <StatisticContainer Container>
auto standard_deviation(const Container& c) {
  return static_cast<typename Container::value_type>(std::sqrt(variance(c)));
}

template <StatisticContainer Container>
auto median(const Container& c) {
  std::vector<typename Container::value_type> values(c.begin(), c.end());
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n % 2 == 0) {
    return (values[n / 2 - 1] + values[n / 2]) / typename Container::value_type(2);
  }
  return values[n / 2];
}

template <StatisticContainer Container>
size_t max_index(const Container& c) {
  if (c.size() == 0) throw std::invalid_argument("max_index of empty container");
  return static_cast<size_t>(std::distance(c.begin(), std::max_element(c.begin(), c.end())));
}

template <StatisticContainer Container>
size_t min_index(const Container& c) {
  if (c.size() == 0) throw std::invalid_argument("min_index of empty container");
  return static_cast<size_t>(std::distance(c.begin(), std::min_element(c.begin(), c.end())));
}

template <StatisticContainer Container>
auto abs_max(const Container& c) {
  if (c.empty()) return typename Container::value_type(0);
  return *std::max_element(c.begin(), c.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); });
}

template <StatisticContainer Container>
auto abs_min(const Container& c) {
  if (c.empty()) return typename Container::value_type(0);
  return *std::min_element(c.begin(), c.end(), [](auto a, auto b) { return std::abs(a) < std::abs(b); });
}

template <typename Derived, typename T>
T sum(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  return detail::arithmetic_range_reduce<false>(expr, 0, expr.size());
}

template <typename Derived, typename T>
T prod(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  return detail::arithmetic_range_reduce<true>(expr, 0, expr.size());
}

template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
bool any(const Container& c) {
  return std::any_of(c.begin(), c.end(), [](const auto& v) { return static_cast<bool>(v); });
}

template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
bool all(const Container& c) {
  return std::all_of(c.begin(), c.end(), [](const auto& v) { return static_cast<bool>(v); });
}

template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
size_t argmin(const Container& c) {
  if (c.size() == 0) throw std::invalid_argument("argmin of empty container");
  return static_cast<size_t>(std::distance(c.begin(), std::min_element(c.begin(), c.end())));
}

template <StatisticContainer Container>
  requires(!detail::md_expression<Container>)
size_t argmax(const Container& c) {
  if (c.size() == 0) throw std::invalid_argument("argmax of empty container");
  return static_cast<size_t>(std::distance(c.begin(), std::max_element(c.begin(), c.end())));
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto sum_axis(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  return detail::arithmetic_axis_reduce<Axis, false>(expr);
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto mean_axis(const base_expr<Derived, T>& expression) {
  if (expression.extents()[Axis] == 0) throw std::invalid_argument("mean_axis of empty axis");
  auto result = sum_axis<Axis>(expression);
  result /= static_cast<std::remove_cv_t<T>>(expression.extents()[Axis]);
  return result;
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto prod_axis(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  return detail::arithmetic_axis_reduce<Axis, true>(expr);
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto min_axis(const base_expr<Derived, T>& expression) {
  if (expression.extents()[Axis] == 0) throw std::invalid_argument("min_axis of empty axis");
  const auto& expr = expression.derived();
  return detail::axis_reduce<Axis, std::remove_cv_t<T>>(expr, [&expr](size_t base, size_t stride, size_t count) {
    if (stride == 1) return detail::extreme_range_reduce<true>(expr, base, count);
    auto result = detail::reduction_value(expr, base);
    for (size_t i = 1; i < count; ++i)
      result = std::min(result, detail::reduction_value(expr, base + i * stride));
    return result;
  });
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto max_axis(const base_expr<Derived, T>& expression) {
  if (expression.extents()[Axis] == 0) throw std::invalid_argument("max_axis of empty axis");
  const auto& expr = expression.derived();
  return detail::axis_reduce<Axis, std::remove_cv_t<T>>(expr, [&expr](size_t base, size_t stride, size_t count) {
    if (stride == 1) return detail::extreme_range_reduce<false>(expr, base, count);
    auto result = detail::reduction_value(expr, base);
    for (size_t i = 1; i < count; ++i)
      result = std::max(result, detail::reduction_value(expr, base + i * stride));
    return result;
  });
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto any_axis(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  return detail::axis_reduce<Axis, int>(expr, [&expr](size_t base, size_t stride, size_t count) {
    if (stride == 1) return detail::logical_range_reduce<true>(expr, base, count);
    for (size_t i = 0; i < count; ++i)
      if (static_cast<bool>(detail::reduction_value(expr, base + i * stride))) return 1;
    return 0;
  });
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto all_axis(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  return detail::axis_reduce<Axis, int>(expr, [&expr](size_t base, size_t stride, size_t count) {
    if (stride == 1) return detail::logical_range_reduce<false>(expr, base, count);
    for (size_t i = 0; i < count; ++i)
      if (!static_cast<bool>(detail::reduction_value(expr, base + i * stride))) return 0;
    return 1;
  });
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto argmin_axis(const base_expr<Derived, T>& expression) {
  if (expression.extents()[Axis] == 0) throw std::invalid_argument("argmin_axis of empty axis");
  const auto& expr = expression.derived();
  return detail::axis_reduce<Axis, size_t>(expr, [&expr](size_t base, size_t stride, size_t count) {
    if (stride == 1) return detail::arg_range_reduce<true>(expr, base, count);
    size_t result = 0;
    auto best = detail::reduction_value(expr, base);
    for (size_t i = 1; i < count; ++i) {
      const auto value = detail::reduction_value(expr, base + i * stride);
      if (value < best) { best = value; result = i; }
    }
    return result;
  });
}

template <size_t Axis, typename Derived, typename T>
  requires(Axis < Derived::rank_ && Derived::rank_ > 1)
auto argmax_axis(const base_expr<Derived, T>& expression) {
  if (expression.extents()[Axis] == 0) throw std::invalid_argument("argmax_axis of empty axis");
  const auto& expr = expression.derived();
  return detail::axis_reduce<Axis, size_t>(expr, [&expr](size_t base, size_t stride, size_t count) {
    if (stride == 1) return detail::arg_range_reduce<false>(expr, base, count);
    size_t result = 0;
    auto best = detail::reduction_value(expr, base);
    for (size_t i = 1; i < count; ++i) {
      const auto value = detail::reduction_value(expr, base + i * stride);
      if (best < value) { best = value; result = i; }
    }
    return result;
  });
}

template <typename Derived, typename T>
T max(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  if (expr.size() == 0) throw std::invalid_argument("max of empty expression");
  std::remove_cv_t<T> result = detail::reduction_value(expr, 0);
  detail::visit_reduction_values(expr, [&result](auto value, size_t index) {
    if (index != 0) result = std::max(result, value);
    return true;
  });
  return result;
}

template <typename Derived, typename T>
T min(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  if (expr.size() == 0) throw std::invalid_argument("min of empty expression");
  std::remove_cv_t<T> result = detail::reduction_value(expr, 0);
  detail::visit_reduction_values(expr, [&result](auto value, size_t index) {
    if (index != 0) result = std::min(result, value);
    return true;
  });
  return result;
}

template <typename Derived, typename T>
bool any(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  bool result = false;
  detail::visit_reduction_values(expr, [&result](auto value, size_t) {
    result = static_cast<bool>(value);
    return !result;
  });
  return result;
}

template <typename Derived, typename T>
bool all(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  bool result = true;
  detail::visit_reduction_values(expr, [&result](auto value, size_t) {
    result = static_cast<bool>(value);
    return result;
  });
  return result;
}

template <typename Derived, typename T>
size_t argmin(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  if (expr.size() == 0) throw std::invalid_argument("argmin of empty expression");
  size_t result = 0;
  auto best = detail::reduction_value(expr, 0);
  detail::visit_reduction_values(expr, [&best, &result](auto value, size_t index) {
    if (index != 0 && value < best) { best = value; result = index; }
    return true;
  });
  return result;
}

template <typename Derived, typename T>
size_t argmax(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  if (expr.size() == 0) throw std::invalid_argument("argmax of empty expression");
  size_t result = 0;
  auto best = detail::reduction_value(expr, 0);
  detail::visit_reduction_values(expr, [&best, &result](auto value, size_t index) {
    if (index != 0 && best < value) { best = value; result = index; }
    return true;
  });
  return result;
}

template <typename Derived, typename T>
T mean(const base_expr<Derived, T>& expression) {
  if (expression.size() == 0) throw std::invalid_argument("mean of empty expression");
  return sum(expression) / static_cast<T>(expression.size());
}

template <typename Derived, typename T>
T variance(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  const T m = mean(expression);
  long double sum_sq = 0.0L;
  for (size_t i = 0; i < expr.size(); ++i) {
    const long double diff = static_cast<long double>(expr[i]) - static_cast<long double>(m);
    sum_sq += diff * diff;
  }
  return static_cast<T>(sum_sq / (expr.size() - 1));
}

template <typename Derived, typename T>
T standard_deviation(const base_expr<Derived, T>& expression) {
  return static_cast<T>(std::sqrt(variance(expression)));
}

template <typename Derived, typename T>
T median(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  std::vector<T> values(expr.size());
  for (size_t i = 0; i < expr.size(); ++i) values[i] = expr[i];
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n % 2 == 0) return (values[n / 2 - 1] + values[n / 2]) / T(2);
  return values[n / 2];
}

template <typename Derived, typename T>
size_t max_index(const base_expr<Derived, T>& expression) {
  return argmax(expression);
}

template <typename Derived, typename T>
size_t min_index(const base_expr<Derived, T>& expression) {
  return argmin(expression);
}

template <typename Derived, typename T>
T abs_max(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  if (expr.size() == 0) return T{};
  size_t result = 0;
  for (size_t i = 1; i < expr.size(); ++i) {
    if (std::abs(expr[result]) < std::abs(expr[i])) result = i;
  }
  return expr[result];
}

template <typename Derived, typename T>
T abs_min(const base_expr<Derived, T>& expression) {
  const auto& expr = expression.derived();
  if (expr.size() == 0) return T{};
  size_t result = 0;
  for (size_t i = 1; i < expr.size(); ++i) {
    if (std::abs(expr[i]) < std::abs(expr[result])) result = i;
  }
  return expr[result];
}

}  // namespace md
