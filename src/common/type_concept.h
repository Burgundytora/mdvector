#ifndef __MDVECTOR_TYPE_CONCEPT__
#define __MDVECTOR_TYPE_CONCEPT__

#include <concepts>
#include <type_traits>

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
  a * b;
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

#endif  //__MDVECTOR_TYPE_CONCEPT__