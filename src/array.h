#ifndef __MDVECTOR_MDARRAY_H__
#define __MDVECTOR_MDARRAY_H__

#include "common/detail.h"
#include "common/iterator_mixin.h"
#include "common/type_concept.h"
#include "expression_template/operator.h"
#include "simd/simd_function.h"
#include "math_function.h"

namespace md {

template <typename T, typename Layout = std::layout_right, size_t... lengths>
class array : public md::tensor_expr<array<T, Layout, lengths...>, T>,
              public md::iterator_mixin<array<T, Layout, lengths...>, T> {
 public:
  using Policy = md::aligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = sizeof...(lengths);

 private:
  /// 成员变量
  static constexpr size_t raw_total_size = (lengths * ... * 1);
  static constexpr size_t total_size = (raw_total_size % md::simd<T>::pack_size == 0)
                                           ? raw_total_size
                                           : ((raw_total_size / md::simd<T>::pack_size) + 1) * md::simd<T>::pack_size;
  alignas(md::simd<T>::alignment) std::array<T, total_size> array_;
  static constexpr std::array<std::size_t, sizeof...(lengths)> shape_ = {static_cast<std::size_t>(lengths)...};
  std::mdspan<T, std::extents<std::size_t, lengths...>, Layout> mdspan_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  explicit array() : mdspan_(array_.data()) {}

  explicit array(T val) : mdspan_(array_.data()) { fill(val); }

  ~array() = default;

  array(const array& other) : array_(other.array_), mdspan_(array_.data()) {}

  array(array&& other) noexcept : array_(std::move(other.array_)), mdspan_(array_.data()) {}

  array& operator=(const array& other) {
    if (this != &other) {
      array_ = other.array_;
      mdspan_ = std::mdspan<T, std::extents<std::size_t, lengths...>, Layout>(array_.data());
    }
    return *this;
  }

  array& operator=(array&& other) noexcept {
    if (this != &other) {
      array_ = std::move(other.array_);
      mdspan_ = std::mdspan<T, std::extents<std::size_t, lengths...>, Layout>(array_.data());
    }
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 多维索引
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& operator[](Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& at(Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 索引转换
  template <typename... Indices>
  size_t get_1d_index(Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    // 使用 mdspan 的 mapping 来获取线性索引
    return mdspan_.mapping()(indices...);
  }

  // 一维索引转多维索引
  std::array<size_t, sizeof...(lengths)> get_md_index(size_t linear_index) const {
    if (linear_index >= raw_total_size) {
      throw std::out_of_range("Linear index out of range");
    }

    std::array<size_t, sizeof...(lengths)> indices{};

    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      // 行优先布局 (C-style)
      size_t remaining = linear_index;
      for (int i = sizeof...(lengths) - 1; i >= 0; --i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
      }
    } else if constexpr (std::is_same_v<Layout, std::layout_left>) {
      // 列优先布局 (Fortran-style)
      size_t remaining = linear_index;
      for (int i = 0; i < sizeof...(lengths); ++i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
      }
    } else {
      // 通用布局，使用 mdspan 的映射器
      auto extents = mdspan_.extents();
      for (int i = 0; i < sizeof...(lengths); ++i) {
        indices[i] = mdspan_.mapping().template operator()<std::size_t>(linear_index, i);
      }
    }

    return indices;
  }

  // 获取指定维度的索引
  size_t get_dim_index(size_t linear_index, size_t dim) const {
    if (linear_index >= raw_total_size || dim >= sizeof...(lengths)) {
      throw std::out_of_range("Index out of range");
    }

    return get_md_index(linear_index)[dim];
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 访问属性
  T* data() { return array_.data(); }

  const T* data() const { return array_.data(); }

  size_t used_size() const { return total_size; }

  size_t size() const { return raw_total_size; }

  auto extents() const { return shape_; }

  size_t extent(int index) const { return shape_.at(index); }

  bool empty() { return false; }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 更改属性
  void fill(T val) { std::fill(array_.begin(), array_.end(), val); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 打印
  void print()
    requires Printable<T>
  {
    md::print_mdspan(mdspan_);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 迭代器
  using md::iterator_mixin<array<T, Layout, lengths...>, T>::begin;
  using md::iterator_mixin<array<T, Layout, lengths...>, T>::end;
  using md::iterator_mixin<array<T, Layout, lengths...>, T>::cbegin;
  using md::iterator_mixin<array<T, Layout, lengths...>, T>::cend;
  using md::iterator_mixin<array<T, Layout, lengths...>, T>::rbegin;
  using md::iterator_mixin<array<T, Layout, lengths...>, T>::rend;
  using md::iterator_mixin<array<T, Layout, lengths...>, T>::crbegin;
  using md::iterator_mixin<array<T, Layout, lengths...>, T>::crend;

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 表达式模板数值计算
  template <typename E>
  array& operator=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    expr.template eval_to<>(*this);
    return *this;
  }

  template <typename T2>
  typename md::simd<T2>::type load_simd(size_t i) const noexcept
    requires Numeric<T>
  {
    return Policy::load<T2>(this->data() + i);
  }

  template <typename T2>
  typename md::simd<T2>::type load_simd_mask(size_t i) const noexcept
    requires Numeric<T>
  {
    return Policy::mask_load<T2>(this->data() + i, used_size() - i);
  }

  template <typename T2>
  typename void store_simd(const size_t& i, md::simd<T2>::const_ref_type simd_val) noexcept {
    return Policy::store<T>(this->data() + i, simd_val);
  }

  template <typename T2>
  typename void store_simd_mask(const size_t& i, const size_t& remaining,
                                md::simd<T2>::const_ref_type simd_val) noexcept {
    return Policy::mask_store<T>(this->data() + i, remaining, simd_val);
  }

  array& operator+=(const array& other) noexcept
    requires Numeric<T>
  {
    md::simd_add_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  array& operator-=(const array& other) noexcept
    requires Numeric<T>
  {
    md::simd_sub_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  array& operator*=(const array& other) noexcept
    requires Numeric<T>
  {
    md::simd_mul_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  array& operator/=(const array& other) noexcept
    requires Numeric<T>
  {
    md::simd_div_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  template <typename E>
  array& operator+=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this + expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  array& operator-=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this - expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  array& operator*=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this * expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  array& operator/=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this / expr).template eval_to<>(*this);
    return *this;
  }

  array& operator+=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_add_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  array& operator-=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_sub_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  array& operator*=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_mul_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  array& operator/=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_div_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  // 取负
  auto operator-() const noexcept
    requires Numeric<T>
  {
    md::vector<T, sizeof...(lengths), Layout> result(this->extents());
    std::transform(this->begin(), this->end(), result.begin(), [](T val) noexcept { return -val; });
    return result;
  }

  // 取正
  auto operator+() const noexcept
    requires Numeric<T>
  {
    return *this;
  }

 private:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 内部函数
  template <typename... Indices>
  void check_indices(Indices... indices) const {
    const size_t idx_array[sizeof...(lengths)] = {static_cast<size_t>(indices)...};
    for (int i = 0; i < sizeof...(lengths); ++i) {
      if (idx_array[i] >= mdspan_.extent(i)) {
        throw std::out_of_range(std::string("Index ") + std::to_string(idx_array[i]) + " out of range for dimension " +
                                std::to_string(i) + " (size: " + std::to_string(mdspan_.extent(i)) + ")");
      }
    }
  }
};

}  // namespace md

// 常用别名
template <typename T, size_t... lengths>
using mdarray_row_major = md::array<T, std::layout_right, lengths...>;

template <typename T, size_t... lengths>
using mdarray_col_major = md::array<T, std::layout_left, lengths...>;

template <typename T, size_t N>
using array_1d = mdarray_row_major<T, N>;

template <typename T, size_t N1, size_t N2>
using array_2d = mdarray_row_major<T, N1, N2>;

template <typename T, size_t N1, size_t N2, size_t N3>
using array_3d = mdarray_row_major<T, N1, N2, N3>;

template <typename T, size_t N1, size_t N2, size_t N3, size_t N4>
using array_4d = mdarray_row_major<T, N1, N2, N3, N4>;

template <typename T, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5>
using array_5d = mdarray_row_major<T, N1, N2, N3, N4, N5>;

template <typename T, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5, size_t N6>
using array_6d = mdarray_row_major<T, N1, N2, N3, N4, N5, N6>;

#endif  // __MDVECTOR_MDARRAY_H__