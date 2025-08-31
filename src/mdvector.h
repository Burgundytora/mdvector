#ifndef __MDVECTOR_H__
#define __MDVECTOR_H__

#include "multi_dimension/engine_dynamic.h"

// base type without simd_ET
template <class T, size_t Rank, class Layout, class Enable>
class mdvector : private md::engine_dynamic<T, Rank, Layout> {
  using Impl = md::engine_dynamic<T, Rank, Layout>;

 public:
  using Impl::Impl;

  mdvector(const mdvector& other) : Impl(other) {}

  mdvector(mdvector&& other) noexcept : Impl(std::move(other)) {}

  mdvector& operator=(const mdvector& other) {
    Impl::operator=(other);
    return *this;
  }

  mdvector& operator=(mdvector&& other) noexcept {
    Impl::operator=(std::move(other));
    return *this;
  }

  ~mdvector() = default;

  using Impl::operator();
#if defined(__cpp_multidimensional_subscript) || __cplusplus >= 202302L
  using Impl::operator[];
#endif
  using Impl::at;
  using Impl::extent;
  using Impl::extents;
  using Impl::reset_shape;
  using Impl::set_value;
  using Impl::shapes;
  using Impl::size;
  using Impl::used_size;

  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  using Impl::begin;
  using Impl::cbegin;
  using Impl::cend;
  using Impl::crbegin;
  using Impl::crend;
  using Impl::end;
  using Impl::rbegin;
  using Impl::rend;
};

// double/float with simd_ET
template <class T, size_t Rank, class Layout>
class mdvector<T, Rank, Layout, std::enable_if_t<std::is_floating_point_v<T>>>
    : public md::tensor_expr<mdvector<T, Rank>, md::unaligned_policy>, private md::engine_dynamic<T, Rank, Layout> {
  using Impl = md::engine_dynamic<T, Rank, Layout>;
  using Policy = md::unaligned_policy;

 public:
  using Impl::Impl;

  mdvector(const mdvector& other) : Impl(other) {}

  mdvector(mdvector&& other) noexcept : Impl(std::move(other)) {}

  mdvector& operator=(const mdvector& other) {
    Impl::operator=(other);
    return *this;
  }

  mdvector& operator=(mdvector&& other) noexcept {
    Impl::operator=(std::move(other));
    return *this;
  }

  ~mdvector() = default;

  using Impl::operator();
#if defined(__cpp_multidimensional_subscript) || __cplusplus >= 202302L
  using Impl::operator[];
#endif
  using Impl::at;
  using Impl::extent;
  using Impl::extents;
  using Impl::reset_shape;
  using Impl::set_value;
  using Impl::shapes;
  using Impl::size;
  using Impl::used_size;

  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  using Impl::begin;
  using Impl::cbegin;
  using Impl::cend;
  using Impl::crbegin;
  using Impl::crend;
  using Impl::end;
  using Impl::rbegin;
  using Impl::rend;

  template <class E>
  mdvector(const md::tensor_expr<E, Policy>& expr) noexcept {
    this->reset_shape(expr.extents());
    expr.eval_to(this->data());
  }

  template <class E>
  mdvector& operator=(const md::tensor_expr<E, Policy>& expr) noexcept {
    this->reset_shape(expr.extents());
    expr.eval_to(this->data());
    return *this;
  }

  // 从span创建
  mdvector(const md::span<T, Rank, Layout>& span) noexcept {
    this->reset_shape(span.extents());
    span.eval_to(this->data());
  }

  mdvector& operator=(const md::span<T, Rank, Layout>& span) noexcept {
    this->reset_shape(span.extents());
    span.eval_to(this->data());
    return *this;
  }

  // 内存连续子视图
  template <class... Slices>
  auto span(Slices... slices) {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");

    constexpr std::size_t NewRank = md::compressed_rank_v<Slices...>;

    auto [slice_array, is_integral] = md::prepare_slices<Rank>(extents(), slices...);

    // 检查越界
    md::check_slice_bounds<Rank>(slice_array, extents());

    // 检查内存连续
    if (!md::check_slice_contiguous<Rank, Layout>(extents(), slice_array, is_integral)) {
      throw std::runtime_error("span slices must result in contiguous memory");
    }

    // 计算新的extents
    std::array<std::size_t, NewRank> new_extents;
    std::size_t new_idx = 0;

    for (std::size_t i = 0; i < Rank; ++i) {
      if (!is_integral[i]) {  // 只保留非整数索引的维度
        const auto& s = slice_array[i];
        std::ptrdiff_t start = md::normalize_index(s.start, extent(i));
        std::ptrdiff_t end = md::normalize_index(s.end, extent(i));
        new_extents[new_idx++] = s.is_all ? extent(i) : (end - start + 1);
      }
    }

    // 计算新的数据指针偏移
    std::size_t offset = calculate_offset(slice_array, is_integral);

    // 返回适当维度的span
    if constexpr (NewRank == 0) {
      // 所有维度都是整数索引，返回标量引用
      return data_[offset];
    } else {
      return md::span<T, NewRank, Layout>(data_.data() + offset, new_extents);
    }
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd(size_t i) const noexcept {
    return md::simd<T2>::load(this->data() + i);
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd_mask(size_t i) const noexcept {
    return md::simd<T2>::mask_load(this->data() + i, used_size() - i);
  }

  mdvector& operator+=(const mdvector& other) noexcept {
    md::simd_add_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdvector& operator-=(const mdvector& other) noexcept {
    md::simd_sub_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdvector& operator*=(const mdvector& other) noexcept {
    md::simd_mul_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdvector& operator/=(const mdvector& other) noexcept {
    md::simd_div_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  template <class E>
  mdvector& operator+=(const md::tensor_expr<E, Policy>& expr) noexcept {
    (*this + expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator-=(const md::tensor_expr<E, Policy>& expr) noexcept {
    (*this - expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator*=(const md::tensor_expr<E, Policy>& expr) noexcept {
    (*this * expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator/=(const md::tensor_expr<E, Policy>& expr) noexcept {
    (*this / expr).eval_to(this->data());
    return *this;
  }

  mdvector& operator+=(T scalar) noexcept {
    md::simd_add_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdvector& operator-=(T scalar) noexcept {
    md::simd_sub_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdvector& operator*=(T scalar) noexcept {
    md::simd_mul_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdvector& operator/=(T scalar) noexcept {
    md::simd_div_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  void show_data_array_style() {
    for (const auto& it : this->data_) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void show_data_matrix_style() {
    if (Rank == 0) return;

    const size_t cols = this->extent(Rank - 1);
    const size_t rows = used_size() / cols;

    for (size_t i = 0; i < rows; ++i) {
      const T* row_start = this->data() + i * cols;
      for (size_t j = 0; j < cols; ++j) {
        std::cout << row_start[j] << " ";
      }
      std::cout << "\n";
    }
  }

  using this_type = mdvector;
  // 数学函数简化定义
#define DEFINE_MD_MATH_OP(name, op)                                                                       \
  this_type name() const noexcept {                                                                       \
    this_type res(*this);                                                                                 \
    std::transform(this->begin(), this->end(), res.begin(), [](T val) noexcept { return std::op(val); }); \
    return res;                                                                                           \
  }
  // 三角函数
  DEFINE_MD_MATH_OP(cos, cos);
  DEFINE_MD_MATH_OP(acos, acos);
  DEFINE_MD_MATH_OP(cosh, cosh);
  DEFINE_MD_MATH_OP(sin, sin);
  DEFINE_MD_MATH_OP(asin, asin);
  DEFINE_MD_MATH_OP(sinh, sinh);
  DEFINE_MD_MATH_OP(tan, tan);
  DEFINE_MD_MATH_OP(atan, atan);
  DEFINE_MD_MATH_OP(tanh, tanh);

  // 数学函数
  DEFINE_MD_MATH_OP(abs, abs);
  DEFINE_MD_MATH_OP(sqrt, sqrt);
  DEFINE_MD_MATH_OP(log10, log10);
  DEFINE_MD_MATH_OP(ln, log);

#undef DEFINE_MD_MATH_OP

  this_type exp(T y) const noexcept {
    this_type res(*this);
    std::transform(this->data_.begin(), this->data_.end(), res.data_.begin(),
                   [y](T val) noexcept { return std::pow(y, val); });
    return res;
  }

  this_type pow(T y) const noexcept {
    this_type res(*this);
    std::transform(this->data_.begin(), this->data_.end(), res.data_.begin(),
                   [y](T val) noexcept { return std::pow(val, y); });
    return res;
  }

 private:
  // 计算数据指针偏移
  std::size_t calculate_offset(const std::array<md::slice, Rank>& slices, const std::array<bool, Rank>& is_integral) {
    std::size_t offset = 0;
    std::size_t stride = 1;

    // 按内存布局计算偏移（这里以行优先为例）
    for (int i = Rank - 1; i >= 0; --i) {
      if (!is_integral[i]) {
        offset += slices[i].start * stride;
        stride *= extent(i);
      } else {
        offset += static_cast<std::size_t>(slices[i].start) * stride;
      }
    }

    return offset;
  }
};

// 视图的数学函数返回一个新的mdvector
#define DEFINE_SPAN_MATH_FUNC(name, func)                                                                   \
  template <class T, size_t Rank, class Layout>                                                             \
  mdvector<T, Rank, Layout> md::span<T, Rank, Layout>::name() const noexcept {                              \
    mdvector<T, Rank, Layout> res(this->extents_);                                                          \
    std::transform(this->begin(), this->end(), res.begin(), [](T val) noexcept { return std::func(val); }); \
    return res;                                                                                             \
  }

// 三角函数
DEFINE_SPAN_MATH_FUNC(cos, cos);
DEFINE_SPAN_MATH_FUNC(acos, acos);
DEFINE_SPAN_MATH_FUNC(cosh, cosh);
DEFINE_SPAN_MATH_FUNC(sin, sin);
DEFINE_SPAN_MATH_FUNC(asin, asin);
DEFINE_SPAN_MATH_FUNC(sinh, sinh);
DEFINE_SPAN_MATH_FUNC(tan, tan);
DEFINE_SPAN_MATH_FUNC(atan, atan);
DEFINE_SPAN_MATH_FUNC(tanh, tanh);
DEFINE_SPAN_MATH_FUNC(abs, abs);
DEFINE_SPAN_MATH_FUNC(sqrt, sqrt);
DEFINE_SPAN_MATH_FUNC(log10, log10);
DEFINE_SPAN_MATH_FUNC(ln, log);

#undef DEFINE_SPAN_MATH_FUNC

template <class T, size_t Rank, class Layout>
mdvector<T, Rank, Layout> md::span<T, Rank, Layout>::exp(T y) const noexcept {
  mdvector<T, Rank, Layout> res(this->extents_);
  std::transform(this->data_.begin(), this->data_.end(), res.data_.begin(),
                 [y](T val) noexcept { return std::pow(y, val); });
  return res;
}

template <class T, size_t Rank, class Layout>
mdvector<T, Rank, Layout> md::span<T, Rank, Layout>::pow(T y) const noexcept {
  mdvector<T, Rank, Layout> res(this->extents_);
  std::transform(this->begin(), this->end(), res.begin(), [y](T val) noexcept { return std::pow(val, y); });
  return res;
}

// 类外数学函数
#define DEFINE_MD_MATH_FUNC(name)                          \
  template <class T, size_t Rank, class Layout>            \
  auto name(const mdvector<T, Rank, Layout>& v) noexcept { \
    return v.name();                                       \
  }                                                        \
  template <class T, size_t Rank, class Layout>            \
  auto name(const md::span<T, Rank, Layout>& v) noexcept { \
    return v.name();                                       \
  }

DEFINE_MD_MATH_FUNC(cos)
DEFINE_MD_MATH_FUNC(acos)
DEFINE_MD_MATH_FUNC(cosh)
DEFINE_MD_MATH_FUNC(sin)
DEFINE_MD_MATH_FUNC(asin)
DEFINE_MD_MATH_FUNC(sinh)
DEFINE_MD_MATH_FUNC(tan)
DEFINE_MD_MATH_FUNC(atan)
DEFINE_MD_MATH_FUNC(tanh)
DEFINE_MD_MATH_FUNC(abs);
DEFINE_MD_MATH_FUNC(sqrt);
DEFINE_MD_MATH_FUNC(log10);
DEFINE_MD_MATH_FUNC(ln);

#undef DEFINE_MD_MATH_FUNC

// 常用别名
using shape_1d = std::array<size_t, 1>;
using shape_2d = std::array<size_t, 2>;
using shape_3d = std::array<size_t, 3>;
using shape_4d = std::array<size_t, 4>;
using shape_5d = std::array<size_t, 5>;
using shape_6d = std::array<size_t, 6>;

template <class T>
using vector_1d = mdvector<T, 1>;

template <class T>
using vector_2d = mdvector<T, 2>;

template <class T>
using vector_3d = mdvector<T, 3>;

template <class T>
using vector_4d = mdvector<T, 4>;

template <class T>
using vector_5d = mdvector<T, 5>;

template <class T>
using vector_6d = mdvector<T, 6>;

#endif  // __MDVECTOR_H__