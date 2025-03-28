#ifndef __SIMD_FUNCTION_H__
#define __SIMD_FUNCTION_H__

#include "simd.h"

template <class T>
void simd_add(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    // float
    const typename simd<T>::type va = simd<T>::load(a + i);
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::add(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::add(va, vb));
}

template <class T>
void simd_sub(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    // float
    const typename simd<T>::type va = simd<T>::load(a + i);
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::sub(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::sub(va, vb));
}

template <class T>
void simd_mul(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    // float
    const typename simd<T>::type va = simd<T>::load(a + i);
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::mul(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::mul(va, vb));
}

template <class T>
void simd_div(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    // float
    const typename simd<T>::type va = simd<T>::load(a + i);
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::div(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::div(va, vb));
}

template <class T>
void simd_add_inplace(T* __restrict a, const T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    // float
    const typename simd<T>::type va = simd<T>::load(a + i);
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(a + i, simd<T>::add(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(a + i, remaining, simd<T>::add(va, vb));
}

template <class T>
void simd_sub_inplace(T* __restrict a, const T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    // float
    const typename simd<T>::type va = simd<T>::load(a + i);
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(a + i, simd<T>::sub(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(a + i, remaining, simd<T>::sub(va, vb));
}

template <class T>
void simd_mul_inplace(T* __restrict a, const T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    // float
    const typename simd<T>::type va = simd<T>::load(a + i);
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(a + i, simd<T>::mul(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(a + i, remaining, simd<T>::mul(va, vb));
}

template <class T>
void simd_div_inplace(T* __restrict a, const T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    // float
    const typename simd<T>::type va = simd<T>::load(a + i);
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(a + i, simd<T>::div(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(a + i, remaining, simd<T>::div(va, vb));
}

// ======================== 向量与标量操作 ========================
template <class T>
void simd_add_scalar(const T* __restrict a, T b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = simd<T>::load(a + i);
    simd<T>::store(c + i, simd<T>::add(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::add(va, vb));
}

template <class T>
void simd_sub_scalar(const T* __restrict a, T b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = simd<T>::load(a + i);
    simd<T>::store(c + i, simd<T>::sub(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::sub(va, vb));
}

template <class T>
void simd_mul_scalar(const T* __restrict a, T b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = simd<T>::load(a + i);
    simd<T>::store(c + i, simd<T>::mul(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::mul(va, vb));
}

template <class T>
void simd_div_scalar(const T* __restrict a, T b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = simd<T>::load(a + i);
    simd<T>::store(c + i, simd<T>::div(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::div(va, vb));
}

// ======================== 向量与标量就地操作 ========================
template <class T>
void simd_add_inplace_scalar(T* __restrict a, T b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = simd<T>::load(a + i);
    simd<T>::store(a + i, simd<T>::add(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  simd<T>::mask_store(a + i, remaining, simd<T>::add(va, vb));
}

template <class T>
void simd_sub_inplace_scalar(T* __restrict a, T b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = simd<T>::load(a + i);
    simd<T>::store(a + i, simd<T>::sub(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  simd<T>::mask_store(a + i, remaining, simd<T>::sub(va, vb));
}

template <class T>
void simd_mul_inplace_scalar(T* __restrict a, T b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = simd<T>::load(a + i);
    simd<T>::store(a + i, simd<T>::mul(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  simd<T>::mask_store(a + i, remaining, simd<T>::mul(va, vb));
}

template <class T>
void simd_div_inplace_scalar(T* __restrict a, T b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = simd<T>::load(a + i);
    simd<T>::store(a + i, simd<T>::div(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type va = simd<T>::mask_load(a + i, remaining);
  simd<T>::mask_store(a + i, remaining, simd<T>::div(va, vb));
}

// ======================== 标量与向量操作 ========================
template <class T>
void simd_scalar_add(T a, const T* __restrict b, T* __restrict c, const size_t n) {
  simd_add_scalar(b, a, c, n);  // 复用加法交换律
}

template <class T>
void simd_scalar_sub(T a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type va = simd<T>::set1(a);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::sub(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::sub(va, vb));
}

template <class T>
void simd_scalar_mul(T a, const T* __restrict b, T* __restrict c, const size_t n) {
  simd_mul_scalar(b, a, c, n);  // 复用乘法交换律
}

template <class T>
void simd_scalar_div(T a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type va = simd<T>::set1(a);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type vb = simd<T>::load(b + i);
    simd<T>::store(c + i, simd<T>::div(va, vb));
  }

  size_t remaining = n - i;
  const typename simd<T>::type vb = simd<T>::mask_load(b + i, remaining);
  simd<T>::mask_store(c + i, remaining, simd<T>::div(va, vb));
}

#endif  // __SIMD_FUNCTION_H__