#ifndef __MDVECTOR_TYPE_CONCEPT_H__
#define __MDVECTOR_TYPE_CONCEPT_H__

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
  a* b;
  a / b;
};

#endif  //__MDVECTOR_TYPE_CONCEPT_H__