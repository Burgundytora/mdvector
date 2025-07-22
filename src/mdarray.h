#ifndef HEADER_MDARRAY_HPP_
#define HEADER_MDARRAY_HPP_

#include "multi_dimension/engine_static.h"

template <class T, class Layout, class Enable, size_t... lengths>
class mdarray_base;

// base type without simd_ET
template <class T, class Layout, size_t... lengths>
class mdarray_base<T, Layout, std::enable_if_t<!std::is_floating_point_v<T>>, lengths...>
    : private md::EngineStatic<T, Layout, lengths...> {
  using Impl = md::EngineStatic<T, Layout, lengths...>;

 public:
  using Impl::Impl;

  mdarray_base(const mdarray_base& other) : Impl(other) {}

  mdarray_base(mdarray_base&& other) noexcept : Impl(std::move(other)) {}

  mdarray_base& operator=(const mdarray_base& other) {
    Impl::operator=(other);
    return *this;
  }

  mdarray_base& operator=(mdarray_base&& other) noexcept {
    Impl::operator=(std::move(other));
    return *this;
  }

  ~mdarray_base() = default;

  using Impl::operator();
  using Impl::at;

  using Impl::extents;
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
template <class T, class Layout, size_t... lengths>
class mdarray_base<T, Layout, std::enable_if_t<std::is_floating_point_v<T>>, lengths...>
    : public md::TensorExpr<mdarray_base<T, Layout, void, lengths...>, md::UnalignedPolicy>,
      private md::EngineStatic<T, Layout, lengths...> {
  using Impl = md::EngineStatic<T, Layout, lengths...>;
  using Policy = md::UnalignedPolicy;

 public:
  using Impl::Impl;

  mdarray_base(const mdarray_base& other) : Impl(other) {}

  mdarray_base(mdarray_base&& other) noexcept : Impl(std::move(other)) {}

  mdarray_base& operator=(const mdarray_base& other) {
    Impl::operator=(other);
    return *this;
  }

  mdarray_base& operator=(mdarray_base&& other) noexcept {
    Impl::operator=(std::move(other));
    return *this;
  }

  ~mdarray_base() = default;

  using Impl::operator();
  using Impl::at;
  using Impl::extents;
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

  // 禁止从表达式构造 mdarray需编译时构造
  // template <class E>
  // mdarray_base(const TensorExpr<E, Policy>& expr) {
  //   this->reset_shape(expr.extents());
  //   expr.eval_to(this->data());
  // }

  template <class E>
  mdarray_base& operator=(const TensorExpr<E, Policy>& expr) {
    expr.eval_to(this->data());
    return *this;
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd(size_t i) const {
    return md::simd<T2>::load(this->data() + i);
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd_mask(size_t i) const {
    return md::simd<T2>::mask_load(this->data() + i, size() - i);
  }

  mdarray_base& operator+=(const mdarray_base& other) {
    md::simd_add_inplace<T, Policy>(this->data(), other.data(), this->size());
    return *this;
  }

  mdarray_base& operator-=(const mdarray_base& other) {
    md::simd_sub_inplace<T, Policy>(this->data(), other.data(), this->size());
    return *this;
  }

  mdarray_base& operator*=(const mdarray_base& other) {
    md::simd_mul_inplace<T, Policy>(this->data(), other.data(), this->size());
    return *this;
  }

  mdarray_base& operator/=(const mdarray_base& other) {
    md::simd_div_inplace<T, Policy>(this->data(), other.data(), this->size());
    return *this;
  }

  template <class E>
  mdarray_base& operator+=(const md::TensorExpr<E, Policy>& expr) {
    (*this + expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdarray_base& operator-=(const md::TensorExpr<E, Policy>& expr) {
    (*this - expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdarray_base& operator*=(const md::TensorExpr<E, Policy>& expr) {
    (*this * expr).eval_to(this->data());
    return *this;
  }

  template <class E>
  mdarray_base& operator/=(const md::TensorExpr<E, Policy>& expr) {
    (*this / expr).eval_to(this->data());
    return *this;
  }

  mdarray_base& operator+=(T scalar) {
    md::simd_add_inplace_scalar<T, Policy>(this->data(), scalar, this->size());
    return *this;
  }

  mdarray_base& operator-=(T scalar) {
    md::simd_sub_inplace_scalar<T, Policy>(this->data(), scalar, this->size());
    return *this;
  }

  mdarray_base& operator*=(T scalar) {
    md::simd_mul_inplace_scalar<T, Policy>(this->data(), scalar, this->size());
    return *this;
  }

  mdarray_base& operator/=(T scalar) {
    md::simd_div_inplace_scalar<T, Policy>(this->data(), scalar, this->size());
    return *this;
  }

  void show_data_array_style() {
    for (const auto& it : this->data_) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void show_data_matrix_style() {
    if (sizeof...(lengths) == 0) return;

    const size_t cols = this->view_.extent(sizeof...(lengths) - 1);
    const size_t rows = used_size() / cols;

    for (size_t i = 0; i < rows; ++i) {
      const T* row_start = this->data() + i * cols;
      for (size_t j = 0; j < cols; ++j) {
        std::cout << row_start[j] << " ";
      }
      std::cout << "\n";
    }
  }

  using this_type = mdarray_base;
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

#define DEFINE_MDARRAY_MATH_FUNC(name)                                       \
  template <class T, class Layout, size_t... lengths>                        \
  auto name(const mdarray_base<T, Layout, void, lengths...>& arr) noexcept { \
    return arr.name();                                                       \
  }

// 生成无参数学函数
DEFINE_MDARRAY_MATH_FUNC(cos)
DEFINE_MDARRAY_MATH_FUNC(sin)
DEFINE_MDARRAY_MATH_FUNC(tan)
DEFINE_MDARRAY_MATH_FUNC(acos)
DEFINE_MDARRAY_MATH_FUNC(asin)
DEFINE_MDARRAY_MATH_FUNC(atan)
DEFINE_MDARRAY_MATH_FUNC(cosh)
DEFINE_MDARRAY_MATH_FUNC(sinh)
DEFINE_MDARRAY_MATH_FUNC(tanh)
DEFINE_MDARRAY_MATH_FUNC(abs)
DEFINE_MDARRAY_MATH_FUNC(sqrt)
DEFINE_MDARRAY_MATH_FUNC(log10)
DEFINE_MDARRAY_MATH_FUNC(ln)

#undef DEFINE_MDARRAY_MATH_FUNC

// 常用别名
template <class T, size_t... lengths>
using mdarray = mdarray_base<T, md::layout_right, void, lengths...>;

template <class T, size_t N>
using array_1d = mdarray<T, N>;

template <class T, size_t N1, size_t N2>
using array_2d = mdarray<T, N1, N2>;

template <class T, size_t N1, size_t N2, size_t N3>
using array_3d = mdarray<T, N1, N2, N3>;

template <class T, size_t N1, size_t N2, size_t N3, size_t N4>
using array_4d = mdarray<T, N1, N2, N3, N4>;

template <class T, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5>
using array_5d = mdarray<T, N1, N2, N3, N4, N5>;

template <class T, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5, size_t N6>
using array_6d = mdarray<T, N1, N2, N3, N4, N5, N6>;

#endif  // HEADER_MDARRAY_HPP_
