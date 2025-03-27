#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

template <class T>
void hwy_add(const T* __restrict a, const T* __restrict b, T* __restrict res, size_t size) {
  const hn::ScalableTag<T> d;
  const size_t pack_size = hn::Lanes(d);
  size_t i = 0;

  // 主循环处理完整向量
  for (; i + pack_size <= size; i += pack_size) {
    const auto va = hn::Load(d, a + i);
    const auto vb = hn::Load(d, b + i);
    hn::Store(hn::Add(va, vb), d, res + i);
  }

  // 处理剩余元素
  if (i < size) {
    const auto mask = hn::FirstN(d, size - i);
    const auto va = hn::LoadN(d, a + i, size - i);
    const auto vb = hn::LoadN(d, b + i, size - i);
    hn::StoreN(hn::Add(va, vb), d, res + i, size - i);
  }
}

template <class T>
void hwy_sub(const T* __restrict a, const T* __restrict b, T* __restrict res, size_t size) {
  const hn::ScalableTag<T> d;
  const size_t pack_size = hn::Lanes(d);
  size_t i = 0;

  for (; i + pack_size <= size; i += pack_size) {
    const auto va = hn::Load(d, a + i);
    const auto vb = hn::Load(d, b + i);
    hn::Store(hn::Sub(va, vb), d, res + i);
  }

  if (i < size) {
    const auto mask = hn::FirstN(d, size - i);
    const auto va = hn::LoadN(d, a + i, size - i);
    const auto vb = hn::LoadN(d, b + i, size - i);
    hn::StoreN(hn::Sub(va, vb), d, res + i, size - i);
  }
}

template <class T>
void hwy_mul(const T* __restrict a, const T* __restrict b, T* __restrict res, size_t size) {
  const hn::ScalableTag<T> d;
  const size_t pack_size = hn::Lanes(d);
  size_t i = 0;

  for (; i + pack_size <= size; i += pack_size) {
    const auto va = hn::Load(d, a + i);
    const auto vb = hn::Load(d, b + i);
    hn::Store(hn::Mul(va, vb), d, res + i);
  }

  if (i < size) {
    const auto mask = hn::FirstN(d, size - i);
    const auto va = hn::LoadN(d, a + i, size - i);
    const auto vb = hn::LoadN(d, b + i, size - i);
    hn::StoreN(hn::Mul(va, vb), d, res + i, size - i);
  }
}

template <class T>
void hwy_div(const T* __restrict a, const T* __restrict b, T* __restrict res, size_t size) {
  const hn::ScalableTag<T> d;
  const size_t pack_size = hn::Lanes(d);
  size_t i = 0;

  for (; i + pack_size <= size; i += pack_size) {
    const auto va = hn::Load(d, a + i);
    const auto vb = hn::Load(d, b + i);
    hn::Store(hn::Div(va, vb), d, res + i);
  }

  if (i < size) {
    const auto mask = hn::FirstN(d, size - i);
    const auto va = hn::LoadN(d, a + i, size - i);
    const auto vb = hn::LoadN(d, b + i, size - i);
    hn::StoreN(hn::Div(va, vb), d, res + i, size - i);
  }
}