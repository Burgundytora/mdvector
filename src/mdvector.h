#ifndef HEADER_MDVECTOR_HPP_
#define HEADER_MDVECTOR_HPP_

#include "multi_dimension/engine_dynamic.h"

// base type without simd_ET
template <class T, size_t Rank, class Layout = md::layout_right, class Enable = void>
class mdvector : private md::EngineDynamic<T, Rank, Layout> {
  using Impl = md::EngineDynamic<T, Rank, Layout>;

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
  using Impl::at;
  using Impl::extents;
  using Impl::reset_shape;
  using Impl::set_value;
  using Impl::shapes;
  using Impl::size;
  using Impl::used_size;
  using Impl::view;

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
    : public md::TensorExpr<mdvector<T, Rank>, md::AlignedPolicy>, private md::EngineDynamic<T, Rank, Layout> {
  using Impl = md::EngineDynamic<T, Rank, Layout>;
  using Policy = md::AlignedPolicy;

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
  using Impl::at;
  using Impl::extents;
  using Impl::reset_shape;
  using Impl::set_value;
  using Impl::shapes;
  using Impl::size;
  using Impl::used_size;
  using Impl::view;

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
  mdvector(const md::TensorExpr<E, Policy>& expr) noexcept {
    this->reset_shape(expr.extents());
    expr.eval_to(this->data());
  }

  template <class E>
  mdvector& operator=(const md::TensorExpr<E, Policy>& expr) noexcept {
    expr.eval_to(this->data());
    return *this;
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
  mdvector& operator+=(const md::TensorExpr<E, Policy>& expr) noexcept {
    (*this + expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator-=(const md::TensorExpr<E, Policy>& expr) noexcept {
    (*this - expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator*=(const md::TensorExpr<E, Policy>& expr) noexcept {
    (*this * expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator/=(const md::TensorExpr<E, Policy>& expr) noexcept {
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

    const size_t cols = this->view_.extent(Rank - 1);
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
#define DEFINE_MD_MATH_OP(name, op)                                                                             \
  this_type name() const noexcept {                                                                             \
    this_type res(*this);                                                                                       \
    std::transform(data_.begin(), data_.end(), res.data_.begin(), [](T val) noexcept { return std::op(val); }); \
    return res;                                                                                                 \
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
  DEFINE_MD_MATH_OP(ln, ln);
  DEFINE_MD_MATH_OP(pow2, pow2);

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
};

// 视图的数学函数返回一个新的mdvector
#define DEFINE_SUBSPAN_MATH_FUNC(name, func)                                                                \
  template <class T, size_t Rank, class Layout>                                                             \
  mdvector<T, Rank, Layout> subspan<T, Rank, Layout>::name() const noexcept {                               \
    mdvector<T, Rank, Layout> res(this->extents_);                                                          \
    std::transform(this->begin(), this->end(), res.begin(), [](T val) noexcept { return std::func(val); }); \
    return res;                                                                                             \
  }

// 三角函数
DEFINE_SUBSPAN_MATH_FUNC(cos, cos);
DEFINE_SUBSPAN_MATH_FUNC(acos, acos);
DEFINE_SUBSPAN_MATH_FUNC(cosh, cosh);
DEFINE_SUBSPAN_MATH_FUNC(sin, sin);
DEFINE_SUBSPAN_MATH_FUNC(asin, asin);
DEFINE_SUBSPAN_MATH_FUNC(sinh, sinh);
DEFINE_SUBSPAN_MATH_FUNC(tan, tan);
DEFINE_SUBSPAN_MATH_FUNC(atan, atan);
DEFINE_SUBSPAN_MATH_FUNC(tanh, tanh);
DEFINE_SUBSPAN_MATH_FUNC(abs, abs);
DEFINE_SUBSPAN_MATH_FUNC(sqrt, sqrt);
DEFINE_SUBSPAN_MATH_FUNC(log10, log10);
DEFINE_SUBSPAN_MATH_FUNC(ln, ln);
DEFINE_SUBSPAN_MATH_FUNC(pow2, pow2);

#undef DEFINE_SUBSPAN_MATH_FUNC

template <class T, size_t Rank, class Layout>
mdvector<T, Rank, Layout> subspan<T, Rank, Layout>::exp(T y) const noexcept {
  mdvector<T, Rank, Layout> res(this->extents_);
  std::transform(this->data_.begin(), this->data_.end(), res.data_.begin(),
                 [y](T val) noexcept { return std::pow(y, val); });
  return res;
}

template <class T, size_t Rank, class Layout>
mdvector<T, Rank, Layout> subspan<T, Rank, Layout>::pow(T y) const noexcept {
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
  auto name(const subspan<T, Rank, Layout>& v) noexcept {  \
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
DEFINE_MD_MATH_FUNC(pow2);

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

#endif  // HEADER_MDVECTOR_HPP_