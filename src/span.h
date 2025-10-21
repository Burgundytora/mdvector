#ifndef __MDVECTOR_SPAN_H__
#define __MDVECTOR_SPAN_H__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "common/detail.h"
#include "common/type_concept.h"
#include "expression_template/operator.h"
#include "simd/simd_function.h"

// 前向声明
template <class T, size_t Rank, class Layout = std::layout_right>
class mdvector;

namespace md {

template <class T, size_t Rank, class Layout = std::layout_right>
class span : public md::tensor_expr<span<T, Rank, Layout>, T> {
  using Policy = md::unaligned_policy;

 protected:
  std::mdspan<T, std::dextents<size_t, Rank>, Layout> mdspan_;
  std::array<size_t, Rank> shape_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  constexpr span() noexcept = default;

  span(T* data, const std::array<std::size_t, Rank>& shape)
      : mdspan_(create_mdspan(data, shape, std::make_index_sequence<Rank>{})), shape_(shape) {}

  span(const span& other) = delete;

  span(const span&& other) = delete;

  span& operator=(const span& other) = delete;

  // 删除移动赋值 不管理所有权
  span& operator=(span&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~span() = default;

  template <class E>
  span(const md::tensor_expr<E, T>& expr) = delete;

  template <class E>
  span& operator=(const md::tensor_expr<E, T>& expr) noexcept {
    expr.template eval_to<T, Policy>(this->data());
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 访问属性
  T* data() { return mdspan_.data_handle(); }

  const T* data() const { return mdspan_.data_handle(); }

  size_t used_size() const noexcept { return this->mdspan_.size(); }

  size_t size() const noexcept { return this->mdspan_.size(); }

  void fill(T val) { std::fill(begin(), end(), val); }

  auto extents() const { return shape_; }

  size_t extent(int index) const { return mdspan_.extent(index); }

  bool empty() { return mdspan_.empty(); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 多维索引
  template <class... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <class... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <class... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <class... Indices>
  const T& operator[](Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <class... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <class... Indices>
  const T& at(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 表达式模板数值计算

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
    (*this + expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  span& operator-=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this - expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  span& operator*=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this * expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  span& operator/=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this / expr).template eval_to<T, Policy>(this->data());
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

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 打印
  void print()
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      md::print_mdspan(mdspan_);
    } else {
      throw std::logic_error("md::span need to be initialized before print!!!");
    }
  }

  using iterator = T*;
  using const_iterator = const T*;

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() noexcept { return data(); }
  iterator end() noexcept { return data() + size(); }
  const_iterator begin() const noexcept { return data(); }
  const_iterator end() const noexcept { return data() + size(); }
  const_iterator cbegin() const noexcept { return data(); }
  const_iterator cend() const noexcept { return data() + size(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

  // 求和
  T sum() { return std::reduce(begin(), end()); }

  // 求积
  T prod() { return std::reduce(begin(), end(), T(1), std::multiplies<T>()); }

  // 最大值
  T max() { return *std::max_element(begin(), end()); }

  // 最小值
  T min() { return *std::min_element(begin(), end()); }

  // 平均值
  T mean() { return std::reduce(begin(), end()) / size(); }

  // 方差
  T variance() {
    if (size() <= 1) {
      return 0.0;
    }
    double m = mean();
    double sum_sq = std::accumulate(begin(), end(), 0.0, [m](double acc, T val) {
      double diff = static_cast<double>(val) - m;
      return acc + diff * diff;
    });

    return sum_sq / (size() - 1);
  }

  // 标准差
  T standard_deviation() { return std::sqrt(variance()); }

  // 中位数
  T median() {
    if (empty()) {
      return 0.0;
    }
    auto vec = std::vector<T>(size());
    std::sort(vec.begin(), vec.end());
    size_t size = vec.size();
    if (size % 2 == 0) {
      return (static_cast<T>(vec[size / 2 - 1]) + static_cast<T>(vec[size / 2])) / 2.0;
    } else {
      return static_cast<T>(vec[size / 2]);
    }
  }

  // 视图的数学函数返回一个新的mdvector
  using return_type = mdvector<T, Rank, Layout>;
  // 三角函数
  return_type cos() const noexcept
    requires Numeric<T>;
  return_type acos() const noexcept
    requires Numeric<T>;
  return_type cosh() const noexcept
    requires Numeric<T>;
  return_type sin() const noexcept
    requires Numeric<T>;
  return_type asin() const noexcept
    requires Numeric<T>;
  return_type sinh() const noexcept
    requires Numeric<T>;
  return_type tan() const noexcept
    requires Numeric<T>;
  return_type atan() const noexcept
    requires Numeric<T>;
  return_type tanh() const noexcept
    requires Numeric<T>;
  // 数学函数
  return_type abs() const noexcept
    requires Numeric<T>;
  return_type exp(T y) const noexcept
    requires Numeric<T>;
  return_type pow(T y) const noexcept
    requires Numeric<T>;
  return_type pow2() const noexcept
    requires Numeric<T>;
  return_type sqrt() const noexcept
    requires Numeric<T>;
  return_type log10() const noexcept
    requires Numeric<T>;
  return_type ln() const noexcept
    requires Numeric<T>;

 private:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 内部函数
  template <size_t... Indices>
  auto create_mdspan(T* data, const std::array<size_t, Rank>& shape, std::index_sequence<Indices...>) {
    return std::mdspan<T, std::dextents<size_t, Rank>, Layout>(data, shape[Indices]...);
  }

  template <typename... Indices>
  void check_indices(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match the rank of mdvector");

    const size_t idx_array[Rank] = {static_cast<size_t>(indices)...};
    for (size_t i = 0; i < Rank; ++i) {
      if (idx_array[i] >= mdspan_.extent(i)) {
        throw std::out_of_range(
            std::format("Index {} out of range for dimension {} (size: {})", idx_array[i], i, mdspan_.extent(i)));
      }
    }
  }
};

}  // namespace md

#endif  // __MDVECTOR_SPAN_H__