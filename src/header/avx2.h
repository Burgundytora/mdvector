#ifndef HEADER_AVX2_H_
#define HEADER_AVX2_H_

#include "allocator.h"

// 1.size小于10000时 显著快于for循环、eigen、mkl
// 2.size在10左右时double 20左右float 有概率很慢
// 3.特大size进行缓存流水线优化

// ========================================================
// 基础运算赋值 c = a ? b
// c = a + b
template <class T>
void avx2_add(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_add_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_add_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// c = a - b
template <class T>
void avx2_sub(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_sub_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_sub_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// c = a * b
template <class T>
void avx2_mul(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_mul_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_mul_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// c = a / b
template <class T>
void avx2_div(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_div_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_div_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// ========================================================
// 基础累积运算 c ?= a
// c += a
template <class T>
void avx2_add_inplace(const T* a, T* c, size_t n) {
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  size_t i = 0;

  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_add_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_add_pd(vc, va));
    }
  }
}

// c -= a
template <class T>
void avx2_sub_inplace(const T* a, T* c, size_t n) {
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  size_t i = 0;

  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_sub_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_sub_pd(vc, va));
    }
  }
}

// c *= a
template <class T>
void avx2_mul_inplace(const T* a, T* c, size_t n) {
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  size_t i = 0;

  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_mul_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_mul_pd(vc, va));
    }
  }
}

// c /= a
template <class T>
void avx2_div_inplace(const T* a, T* c, size_t n) {
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  size_t i = 0;

  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_div_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_div_pd(vc, va));
    }
  }
}

// ========================================================
//  c ?= a

// c = a*b+c
template <class T>
void avx2_fma(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_div_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_div_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// // AVX2赋值实现
// template <typename E>
// void avx2_assign(const E& expr) {
//   constexpr size_t pack_size = SimdConfig<T>::pack_size;
//   const size_t aligned_size = (total_elements_ / pack_size) * pack_size;

//   if constexpr (std::is_same_v<T, float>) {
//     for (size_t i = 0; i < aligned_size; i += pack_size) {
//       _mm256_store_ps(data_ + i, expr[i]);
//     }
//   } else {
//     for (size_t i = 0; i < aligned_size; i += pack_size) {
//       _mm256_store_pd(data_ + i, expr[i]);
//     }
//   }
// }

#endif  // HEADER_AVX2_H_