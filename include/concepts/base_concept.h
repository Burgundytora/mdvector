#ifndef __MDVECTOR_TYPE_CONCEPT__
#define __MDVECTOR_TYPE_CONCEPT__

#include <concepts>
#include <type_traits>
#include <iostream>

// 定义概念
template <typename T>
concept Printable = requires(std::ostream& os, const T& obj) { os << obj; };

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <typename T>
concept HasSwap = requires(T& a, T& b) { a.swap(b); };

template <typename T>
concept Comparable = requires(const T& a, const T& b) {
  { a == b } -> std::convertible_to<bool>;
  { a != b } -> std::convertible_to<bool>;
};

template <typename T>
concept Arithmetic = requires(T a, T b) {
  a + b;
  a - b;
  a* b;
  a / b;
};

// 定义容器概念
template <typename T>
concept MultiDimContainer = requires(T c) {
  { T::rank_ } -> std::convertible_to<size_t>;
};

// 定义容器概念
template <typename T>
concept StatisticContainer = requires(T v) {
  typename T::value_type;
  { v.begin() } -> std::input_iterator;
  { v.end() } -> std::input_iterator;
};

// 通用函数 目前没想好放在哪
#include <array>
template <size_t Rank>
size_t calculate_size(const std::array<size_t, Rank>& shape) {
  return std::reduce(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

#include <format>
#include <array>
template <typename T, size_t N>
struct std::formatter<std::array<T, N>> : std::formatter<std::string> {
  auto format(const std::array<T, N>& arr, format_context& ctx) const {
    std::string result = "[";
    for (size_t i = 0; i < N; ++i) {
      if (i > 0) result += ", ";
      result += std::to_string(arr[i]);
    }
    result += "]";
    return std::formatter<std::string>::format(result, ctx);
  }
};

#endif  //__MDVECTOR_TYPE_CONCEPT__