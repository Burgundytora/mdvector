#ifndef __SIMD_FUNCTION_H__
#define __SIMD_FUNCTION_H__

#include "simd.h"

namespace md {

// ======================== 向量与向量操作 ========================
template <class T, class Policy>
void simd_add(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(c + i, simd<T>::add(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::add(va, vb));
}

template <class T, class Policy>
void simd_sub(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(c + i, simd<T>::sub(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::sub(va, vb));
}

template <class T, class Policy>
void simd_mul(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(c + i, simd<T>::mul(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::mul(va, vb));
}

template <class T, class Policy>
void simd_div(const T* __restrict a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(c + i, simd<T>::div(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::div(va, vb));
}

// ======================== 向量与向量就地操作 ========================
template <class T, class Policy>
void simd_add_inplace(T* __restrict a, const T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(a + i, simd<T>::add(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(a + i, remaining, simd<T>::add(va, vb));
}

template <class T, class Policy>
void simd_sub_inplace(T* __restrict a, const T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(a + i, simd<T>::sub(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(a + i, remaining, simd<T>::sub(va, vb));
}

template <class T, class Policy>
void simd_mul_inplace(T* __restrict a, const T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(a + i, simd<T>::mul(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(a + i, remaining, simd<T>::mul(va, vb));
}

template <class T, class Policy>
void simd_div_inplace(T* __restrict a, const T* __restrict b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(a + i, simd<T>::div(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(a + i, remaining, simd<T>::div(va, vb));
}

// ======================== 向量与标量操作 ========================
template <class T, class Policy>
void simd_add_scalar(const T* __restrict a, T b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    Policy::template store<T>(c + i, simd<T>::add(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::add(va, vb));
}

template <class T, class Policy>
void simd_sub_scalar(const T* __restrict a, T b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    Policy::template store<T>(c + i, simd<T>::sub(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::sub(va, vb));
}

template <class T, class Policy>
void simd_mul_scalar(const T* __restrict a, T b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    Policy::template store<T>(c + i, simd<T>::mul(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::mul(va, vb));
}

template <class T, class Policy>
void simd_div_scalar(const T* __restrict a, T b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    Policy::template store<T>(c + i, simd<T>::div(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::div(va, vb));
}

// ======================== 向量与标量就地操作 ========================
template <class T, class Policy>
void simd_add_inplace_scalar(T* __restrict a, T b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    Policy::template store<T>(a + i, simd<T>::add(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  Policy::template mask_store<T>(a + i, remaining, simd<T>::add(va, vb));
}

template <class T, class Policy>
void simd_sub_inplace_scalar(T* __restrict a, T b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    Policy::template store<T>(a + i, simd<T>::sub(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  Policy::template mask_store<T>(a + i, remaining, simd<T>::sub(va, vb));
}

template <class T, class Policy>
void simd_mul_inplace_scalar(T* __restrict a, T b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    Policy::template store<T>(a + i, simd<T>::mul(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  Policy::template mask_store<T>(a + i, remaining, simd<T>::mul(va, vb));
}

template <class T, class Policy>
void simd_div_inplace_scalar(T* __restrict a, T b, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type vb = simd<T>::set1(b);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type va = Policy::template load<T>(a + i);
    Policy::template store<T>(a + i, simd<T>::div(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type va = Policy::template mask_load<T>(a + i, remaining);
  Policy::template mask_store<T>(a + i, remaining, simd<T>::div(va, vb));
}

// ======================== 标量与向量操作 ========================
template <class T, class Policy>
void simd_scalar_add(T a, const T* __restrict b, T* __restrict c, const size_t n) {
  simd_add_scalar<T, Policy>(b, a, c, n);
}

template <class T, class Policy>
void simd_scalar_sub(T a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type va = simd<T>::set1(a);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(c + i, simd<T>::sub(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::sub(va, vb));
}

template <class T, class Policy>
void simd_scalar_mul(T a, const T* __restrict b, T* __restrict c, const size_t n) {
  simd_mul_scalar<T, Policy>(b, a, c, n);
}

template <class T, class Policy>
void simd_scalar_div(T a, const T* __restrict b, T* __restrict c, const size_t n) {
  constexpr size_t pack_size = simd<T>::pack_size;
  const typename simd<T>::type va = simd<T>::set1(a);

  size_t i = 0;
  for (; i + pack_size <= n; i += pack_size) {
    const typename simd<T>::type vb = Policy::template load<T>(b + i);
    Policy::template store<T>(c + i, simd<T>::div(va, vb));
  }

  const size_t remaining = n - i;
  const typename simd<T>::type vb = Policy::template mask_load<T>(b + i, remaining);
  Policy::template mask_store<T>(c + i, remaining, simd<T>::div(va, vb));
}

}  // namespace md

#endif  // __SIMD_FUNCTION_H__