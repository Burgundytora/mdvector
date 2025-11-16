#ifndef __MDVECTOR_SPAN_H__
#define __MDVECTOR_SPAN_H__

#include "common/detail.h"
#include "common/iterator_mixin.h"
#include "common/math_function.h"
#include "common/statistic_function.h"
#include "common/type_concept.h"
#include "expression_template/operator.h"
#include "simd/simd_function.h"

namespace md {

template <typename T, size_t Rank, typename Layout = std::layout_right>
class span : public md::tensor_expr<span<T, Rank, Layout>, T>, public md::iterator_mixin<span<T, Rank, Layout>, T> {
 public:
  using Policy = md::unaligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

 protected:
  std::mdspan<T, std::dextents<size_t, Rank>, Layout> mdspan_;
  std::array<size_t, Rank> shape_;
  size_t size_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  constexpr span() noexcept = default;

  span(T* data, const std::array<std::size_t, Rank>& shape)
      : mdspan_(create_mdspan(data, shape, std::make_index_sequence<Rank>{})),
        shape_(shape),
        size_(md::calculate_size(shape)) {}

  span(const span& other) = delete;

  span(const span&& other) = delete;

  span& operator=(const span& other) = delete;

  // 删除移动赋值 不管理所有权
  span& operator=(span&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~span() = default;

  template <typename E>
  span(const md::tensor_expr<E, T>& expr) = delete;

  template <typename E>
  span& operator=(const md::tensor_expr<E, T>& expr) noexcept {
    expr.template eval_to<T, Policy>(this->data());
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 访问属性
  T* data() { return mdspan_.data_handle(); }

  const T* data() const { return mdspan_.data_handle(); }

  size_t used_size() const noexcept { return size_; }

  size_t size() const noexcept { return size_; }

  void fill(T val) { std::fill(begin(), end(), val); }

  auto extents() const { return shape_; }

  size_t extent(int index) const { return shape_.at(index); }

  bool empty() { return mdspan_.empty(); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 多维索引
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& operator[](Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& at(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 索引转换
  template <typename... Indices>
  size_t get_1d_index(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    // 使用 mdspan 的 mapping 来获取线性索引
    return mdspan_.mapping()(indices...);
  }

  // 一维索引转多维索引
  std::array<size_t, Rank> get_md_index(size_t linear_index) const {
    if (linear_index >= size_) {
      throw std::out_of_range("Linear index out of range");
    }

    std::array<size_t, Rank> indices{};

    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      // 行优先布局 (C-style)
      size_t remaining = linear_index;
      for (int i = Rank - 1; i >= 0; --i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
      }
    } else if constexpr (std::is_same_v<Layout, std::layout_left>) {
      // 列优先布局 (Fortran-style)
      size_t remaining = linear_index;
      for (int i = 0; i < Rank; ++i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
      }
    } else {
      // 通用布局，使用 mdspan 的映射器
      auto extents = mdspan_.extents();
      for (int i = 0; i < Rank; ++i) {
        indices[i] = mdspan_.mapping().template operator()<std::size_t>(linear_index, i);
      }
    }

    return indices;
  }

  // 获取指定维度的索引
  size_t get_dim_index(size_t linear_index, size_t dim) const {
    if (linear_index >= size_ || dim >= Rank) {
      throw std::out_of_range("Index out of range");
    }

    return get_md_index(linear_index)[dim];
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 表达式模板数值计算

  template <typename T2>
  typename md::simd<T2>::type eval_simd(size_t i) const noexcept {
    return md::simd<T2>::loadu(this->data() + i);
  }

  template <typename T2>
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

  template <typename E>
  span& operator+=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this + expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <typename E>
  span& operator-=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this - expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <typename E>
  span& operator*=(const md::tensor_expr<E, T>& expr) noexcept {
    (*this * expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <typename E>
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

  // 取负
  auto operator-() const noexcept
    requires Numeric<T>
  {
    mdvector<T, Rank, Layout> result(this->extents());
    std::transform(this->begin(), this->end(), result.begin(), [](T val) noexcept { return -val; });
    return result;
  }

  // 取正
  auto operator+() const noexcept
    requires Numeric<T>
  {
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

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 迭代器
  using md::iterator_mixin<span<T, Rank, Layout>, T>::begin;
  using md::iterator_mixin<span<T, Rank, Layout>, T>::end;
  using md::iterator_mixin<span<T, Rank, Layout>, T>::cbegin;
  using md::iterator_mixin<span<T, Rank, Layout>, T>::cend;
  using md::iterator_mixin<span<T, Rank, Layout>, T>::rbegin;
  using md::iterator_mixin<span<T, Rank, Layout>, T>::rend;
  using md::iterator_mixin<span<T, Rank, Layout>, T>::crbegin;
  using md::iterator_mixin<span<T, Rank, Layout>, T>::crend;

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
    for (int i = 0; i < Rank; ++i) {
      if (idx_array[i] >= mdspan_.extent(i)) {
        throw std::out_of_range(
            std::format("Index {} out of range for dimension {} (size: {})", idx_array[i], i, mdspan_.extent(i)));
      }
    }
  }
};

}  // namespace md

#endif  // __MDVECTOR_SPAN_H__