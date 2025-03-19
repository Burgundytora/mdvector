#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

template <class T>
void hwy_add(const T* __restrict a, const T* __restrict b, T* __restrict res, size_t size) {
  const hn::ScalableTag<T> d;
  for (size_t i = 0; i <= size - hn::Lanes(d); i += hn::Lanes(d)) {
    const auto va = hn::Load(d, a + i);
    const auto vb = hn::Load(d, b + i);
    hn::Store(hn::Add(va, vb), d, res + i);
  }
}

template <class T>
void hwy_sub(const T* a, const T* b, T* res, size_t size) {
  const hn::ScalableTag<T> d;
  for (size_t i = 0; i < size; i += hn::Lanes(d)) {
    const auto va = hn::Load(d, a + i);
    const auto vb = hn::Load(d, b + i);
    hn::Store(hn::Sub(va, vb), d, res + i);
  }
}

template <class T>
void hwy_mul(const T* a, const T* b, T* res, size_t size) {
  const hn::ScalableTag<T> d;
  for (size_t i = 0; i < size; i += hn::Lanes(d)) {
    const auto va = hn::Load(d, a + i);
    const auto vb = hn::Load(d, b + i);
    hn::Store(hn::Mul(va, vb), d, res + i);
  }
}

template <class T>
void hwy_div(const T* a, const T* b, T* res, size_t size) {
  const hn::ScalableTag<T> d;
  for (size_t i = 0; i < size; i += hn::Lanes(d)) {
    const auto va = hn::Load(d, a + i);
    const auto vb = hn::Load(d, b + i);
    hn::Store(hn::Div(va, vb), d, res + i);
  }
}