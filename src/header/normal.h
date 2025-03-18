#ifndef HEADER_NORMAL_H_
#define HEADER_NORMAL_H_

#include <type_traits>

using std::size_t;

template <class T>
void norm_add(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  for (size_t i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void norm_sub(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  for (size_t i = 0; i < n; i++) {
    c[i] = a[i] - b[i];
  }
}

template <class T>
void norm_mul(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  for (size_t i = 0; i < n; i++) {
    c[i] = a[i] * b[i];
  }
}

template <class T>
void norm_div(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  for (size_t i = 0; i < n; i++) {
    c[i] = a[i] / b[i];
  }
}

#endif  // HEADER_NORMAL_H_