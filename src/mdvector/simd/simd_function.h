#ifndef __SIMD_FUNCTION_H__
#define __SIMD_FUNCTION_H__

#include "simd.h"

template <class T>
void simd_add(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    // float
    const simd<T>::type va = simd<T>::load(a + i);
    const simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::add(va, vb));
  }
  size_t remaining = n - i;
  if (remaining > 0) {
    const simd<T>::type va = simd<T>::mask_load(a + i, remaining);
    const simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
    simd<T>::mask_store(c + i, remaining, simd<T>::add(va, vb));
  }
}

template <class T>
void simd_sub(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    // float
    const simd<T>::type va = simd<T>::load(a + i);
    const simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::sub(va, vb));
  }
  size_t remaining = n - i;
  if (remaining > 0) {
    const simd<T>::type va = simd<T>::mask_load(a + i, remaining);
    const simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
    simd<T>::mask_store(c + i, remaining, simd<T>::sub(va, vb));
  }
}

template <class T>
void simd_mul(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    // float
    const simd<T>::type va = simd<T>::load(a + i);
    const simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::mul(va, vb));
  }
  size_t remaining = n - i;
  if (remaining > 0) {
    const simd<T>::type va = simd<T>::mask_load(a + i, remaining);
    const simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
    simd<T>::mask_store(c + i, remaining, simd<T>::mul(va, vb));
  }
}

template <class T>
void simd_div(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    // float
    const simd<T>::type va = simd<T>::load(a + i);
    const simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::div(va, vb));
  }
  size_t remaining = n - i;
  if (remaining > 0) {
    const simd<T>::type va = simd<T>::mask_load(a + i, remaining);
    const simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
    simd<T>::mask_store(c + i, remaining, simd<T>::div(va, vb));
  }
}

template <class T>
void simd_add_inplace(const T* __restrict a, T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    // float
    const simd<T>::type va = simd<T>::load(a + i);
    const simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(b + i, simd<T>::add(va, vb));
  }
  size_t remaining = n - i;
  if (remaining > 0) {
    const simd<T>::type va = simd<T>::mask_load(a + i, remaining);
    const simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
    simd<T>::mask_store(b + i, remaining, simd<T>::add(va, vb));
  }
}

template <class T>
void simd_sub_inplace(const T* __restrict a, T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    // float
    const simd<T>::type va = simd<T>::load(a + i);
    const simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(b + i, simd<T>::sub(va, vb));
  }
  size_t remaining = n - i;
  if (remaining > 0) {
    const simd<T>::type va = simd<T>::mask_load(a + i, remaining);
    const simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
    simd<T>::mask_store(b + i, remaining, simd<T>::sub(va, vb));
  }
}

template <class T>
void simd_mul_inplace(const T* __restrict a, T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    // float
    const simd<T>::type va = simd<T>::load(a + i);
    const simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(b + i, simd<T>::mul(va, vb));
  }
  size_t remaining = n - i;
  if (remaining > 0) {
    const simd<T>::type va = simd<T>::mask_load(a + i, remaining);
    const simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
    simd<T>::mask_store(b + i, remaining, simd<T>::mul(va, vb));
  }
}

template <class T>
void simd_div_inplace(const T* __restrict a, T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    // float
    const simd<T>::type va = simd<T>::load(a + i);
    const simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(b + i, simd<T>::div(va, vb));
  }
  size_t remaining = n - i;
  if (remaining > 0) {
    const simd<T>::type va = simd<T>::mask_load(a + i, remaining);
    const simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
    simd<T>::mask_store(b + i, remaining, simd<T>::div(va, vb));
  }
}

#endif  // __SIMD_FUNCTION_H__