#ifndef __MDVECTOR_SPAN_SUBSPAN_H__
#define __MDVECTOR_SPAN_SUBSPAN_H__

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "mdspan.h"

// 前向声明 用于subspan数学函数返回mdvector
template <class T, size_t Rank, class Layout = md::layout_right, class Enable = void>
class mdvector;

template <class T, size_t Rank, class Layout = md::layout_right>
class subspan : public mdspan<T, Rank, Layout>, public md::TensorExpr<subspan<T, Rank, Layout>, md::UnalignedPolicy> {
  using Policy = md::UnalignedPolicy;

 public:
  constexpr subspan() noexcept = default;

  subspan(T* data, const std::array<std::size_t, Rank>& extents, const std::array<md::slice, Rank>& slice_set)
      : mdspan<T, Rank, Layout>(nullptr, {}) {
    md::check_slice_bounds<Rank>(slice_set, extents);
    if (!is_contiguous_slice(slice_set)) {
      throw std::runtime_error("subspan slices must result in contiguous memory");
    }

    std::array<std::size_t, Rank> new_extents;
    std::array<std::size_t, Rank> new_strides = md::compute_strides(extents);
    std::size_t offset = 0;

    for (size_t i = 0; i < Rank; ++i) {
      if (slice_set[i].is_all) {
        new_extents[i] = extents[i];
      } else {
        // 处理负数索引
        std::ptrdiff_t start = md::normalize_index(slice_set[i].start, extents[i]);
        std::ptrdiff_t end = md::normalize_index(slice_set[i].end, extents[i]);

        new_extents[i] = end - start + 1;  // 闭区间大小
        offset += start * new_strides[i];  // 计算内存偏移
      }
    }

    this->data_ = data + offset;
    this->extents_ = new_extents;
    this->strides_ = new_strides;

    this->size_ = std::accumulate(new_extents.begin(), new_extents.end(), size_t(1), std::multiplies<>());
  }

  subspan(const subspan& other) = default;

  // 禁止移动构造
  subspan(const subspan&& other) = delete;

  // ---------- 赋值运算符 ----------
  // 拷贝赋值：将右侧数据复制到当前视图（不修改指针和大小）
  subspan& operator=(const subspan& other) noexcept {
    std::copy(other.begin(), other.end(), this->data_);  // 逐元素复制
    return *this;
  }

  // 删除移动赋值 不管理所有权
  subspan& operator=(subspan&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~subspan() = default;

  size_t used_size() const noexcept { return this->size_; }

  size_t size() const noexcept { return this->size_; }

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

  template <class E>
  subspan(const md::TensorExpr<E, Policy>& expr) = delete;

  template <class E>
  subspan& operator=(const md::TensorExpr<E, Policy>& expr) noexcept {
    expr.eval_to(this->data());
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

  subspan& operator+=(const subspan& other) noexcept {
    simd_add_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  subspan& operator-=(const subspan& other) noexcept {
    simd_sub_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  subspan& operator*=(const subspan& other) noexcept {
    simd_mul_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  subspan& operator/=(const subspan& other) noexcept {
    simd_div_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  template <class E>
  subspan& operator+=(const md::TensorExpr<E, Policy>& expr) noexcept {
    (*this + expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  subspan& operator-=(const md::TensorExpr<E, Policy>& expr) noexcept {
    (*this - expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  subspan& operator*=(const md::TensorExpr<E, Policy>& expr) noexcept {
    (*this * expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  subspan& operator/=(const md::TensorExpr<E, Policy>& expr) noexcept {
    (*this / expr).eval_to(this->data());
    return *this;
  }

  subspan& operator+=(T scalar) noexcept {
    md::simd_add_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  subspan& operator-=(T scalar) noexcept {
    md::simd_sub_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  subspan& operator*=(T scalar) noexcept {
    md::simd_mul_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  subspan& operator/=(T scalar) noexcept {
    md::simd_div_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

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
  // 检查切片是否满足内存连续
  bool is_contiguous_slice(const std::array<md::slice, Rank>& slices) {
    for (int i = Rank - 1; i >= 0; --i) {
      if (!slices[i].is_all) {
        for (int j = i - 1; j >= 0; j--) {
          // 单元素检查改为 start == end
          if (slices[j].start != slices[j].end) {
            return false;
          }
        }
        break;
      }
    }
    return true;
  }
};

#endif  // MDVECTOR_SPAN_SUBSPAN_H_