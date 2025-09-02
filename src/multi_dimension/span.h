#ifndef __MDVECTOR_SPAN_H__
#define __MDVECTOR_SPAN_H__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "mdspan.h"

template <class T, size_t Rank, class Layout = md::layout_right, class Enable = void>
class mdvector;

namespace md {

template <class T, size_t Rank, class Layout = md::layout_right>
class span : public mdspan<T, Rank, Layout>, public md::tensor_expr<span<T, Rank, Layout>, T> {
  using Policy = md::unaligned_policy;

 public:
  constexpr span() noexcept = default;

  using mdspan::extents;
  using mdspan::mdspan;

  span(T* data, const std::array<std::size_t, Rank>& extents) : mdspan<T, Rank, Layout>(data, extents) {
    this->size_ = std::accumulate(extents.begin(), extents.end(), size_t(1), std::multiplies<>());
  }

  span(const span& other) = default;

  span(const span&& other) = delete;

  // ---------- 赋值运算符 ----------
  // 拷贝赋值：将右侧数据复制到当前视图（不修改指针和大小）
  span& operator=(const span& other) = delete;

  // 删除移动赋值 不管理所有权
  span& operator=(span&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~span() = default;

  template <class E>
  span(const md::tensor_expr<E, T>& expr) = delete;

  template <class E>
  span& operator=(const md::tensor_expr<E, T>& expr) noexcept {
    expr.eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd(size_t i) const noexcept {
    return md::simd<T2>::loadu(this->data() + i);
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd_mask(size_t i) const noexcept {
    return md::simd<T2>::mask_loadu(this->data() + i, this->used_size() - i);
  }

  span& operator+=(const span& other) noexcept {
    simd_add_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  span& operator-=(const span& other) noexcept {
    simd_sub_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  span& operator*=(const span& other) noexcept {
    simd_mul_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  span& operator/=(const span& other) noexcept {
    simd_div_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  template <class E>
  span& operator+=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this + expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  span& operator-=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this - expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  span& operator*=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this * expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  span& operator/=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this / expr).eval_to(this->data());
    return *this;
  }

  span& operator+=(T scalar) noexcept {
    md::simd_add_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  span& operator-=(T scalar) noexcept {
    md::simd_sub_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  span& operator*=(T scalar) noexcept {
    md::simd_mul_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  span& operator/=(T scalar) noexcept {
    md::simd_div_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  size_t used_size() const noexcept { return this->size_; }

  size_t size() const noexcept { return this->size_; }

  void set_value(T val) { std::fill(begin(), end(), val); }

  void show_data_array_style() {
    for (const auto& it : *this) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void show_data_matrix_style() {
    if (Rank == 0) return;

    const size_t cols = this->extents_[Rank - 1];
    const size_t rows = used_size() / cols;

    for (size_t i = 0; i < rows; ++i) {
      const T* row_start = this->data() + i * cols;
      for (size_t j = 0; j < cols; ++j) {
        std::cout << row_start[j] << " ";
      }
      std::cout << "\n";
    }
  }

  using iterator = T*;
  using const_iterator = const T*;

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() noexcept { return this->data_; }
  iterator end() noexcept { return this->data_ + this->size_; }
  const_iterator begin() const noexcept { return this->data_; }
  const_iterator end() const noexcept { return this->data_ + this->size_; }
  const_iterator cbegin() const noexcept { return this->data_; }
  const_iterator cend() const noexcept { return this->data_ + this->size_; }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

  // 视图的数学函数返回一个新的mdvector
  using return_type = mdvector<T, Rank, Layout>;
  // 三角函数
  return_type cos() const noexcept;
  return_type acos() const noexcept;
  return_type cosh() const noexcept;
  return_type sin() const noexcept;
  return_type asin() const noexcept;
  return_type sinh() const noexcept;
  return_type tan() const noexcept;
  return_type atan() const noexcept;
  return_type tanh() const noexcept;
  // 数学函数
  return_type abs() const noexcept;
  return_type exp(T y) const noexcept;
  return_type pow(T y) const noexcept;
  return_type pow2() const noexcept;
  return_type sqrt() const noexcept;
  return_type log10() const noexcept;
  return_type ln() const noexcept;

 private:
};

}  // namespace md

#endif  // __MDVECTOR_SPAN_H__