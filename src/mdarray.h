#ifndef __MDVECTOR_ENGINE_STATIC_H__
#define __MDVECTOR_ENGINE_STATIC_H__

#include "common/detail.h"
#include "common/type_concept.h"
#include "expression_template/operator.h"
#include "simd/simd_function.h"

template <class T, class Layout = std::layout_right, size_t... lengths>
class mdarray : public md::tensor_expr<mdarray<T, Layout, lengths...>, T> {
  /// simd对齐策略
  using Policy = md::aligned_policy;

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
  explicit mdarray() : mdspan_(array_.data()) {}

  ~mdarray() = default;

  mdarray(const mdarray& other) : array_(other.array_), mdspan_(array_.data()) {}

  mdarray(mdarray&& other) noexcept : array_(std::move(other.array_)), mdspan_(array_.data()) {}

  mdarray& operator=(const mdarray& other) {
    if (this != &other) {
      array_ = other.array_;
      mdspan_ = std::mdspan<T, std::extents<std::size_t, lengths...>, Layout>(array_.data());
    }
    return *this;
  }

  mdarray& operator=(mdarray&& other) noexcept {
    if (this != &other) {
      array_ = std::move(other.array_);
      mdspan_ = std::mdspan<T, std::extents<std::size_t, lengths...>, Layout>(array_.data());
    }
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 多维索引
  template <class... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <class... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <class... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <class... Indices>
  const T& operator[](Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <class... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <class... Indices>
  const T& at(Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <class... Indices>
  size_t get_1d_index(Indices... indices) const {
    static_assert(sizeof...(Indices) == sizeof...(lengths), "Number of indices must match rank");
    // 使用 mdspan 的 mapping 来获取线性索引
    return mdspan_.mapping()(indices...);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 访问属性
  T* data() { return array_.data(); }

  const T* data() const { return array_.data(); }

  size_t used_size() const { return array_.size(); }

  size_t size() const { return raw_total_size; }

  auto extents() const { return shape_; }

  size_t extent(int i) const { return mdspan_.extent(i); }

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
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() noexcept { return array_.begin(); }
  iterator end() noexcept { return array_.begin() + raw_total_size; }
  const_iterator begin() const noexcept { return array_.begin(); }
  const_iterator end() const noexcept { return array_.begin() + raw_total_size; }
  const_iterator cbegin() const noexcept { return array_.begin(); }
  const_iterator cend() const noexcept { return array_.begin() + raw_total_size; }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 表达式模板数值计算
  template <class E>
  mdarray& operator=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    expr.template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd(size_t i) const noexcept
    requires Numeric<T>
  {
    return md::simd<T2>::load(this->data() + i);
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd_mask(size_t i) const noexcept
    requires Numeric<T>
  {
    return md::simd<T2>::mask_load(this->data() + i, used_size() - i);
  }

  mdarray& operator+=(const mdarray& other) noexcept
    requires Numeric<T>
  {
    md::simd_add_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdarray& operator-=(const mdarray& other) noexcept
    requires Numeric<T>
  {
    md::simd_sub_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdarray& operator*=(const mdarray& other) noexcept
    requires Numeric<T>
  {
    md::simd_mul_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdarray& operator/=(const mdarray& other) noexcept
    requires Numeric<T>
  {
    md::simd_div_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  template <class E>
  mdarray& operator+=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this + expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  mdarray& operator-=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this - expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  mdarray& operator*=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this * expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  mdarray& operator/=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this / expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  mdarray& operator+=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_add_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdarray& operator-=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_sub_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdarray& operator*=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_mul_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdarray& operator/=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_div_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  ///
  using this_type = mdarray;
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
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 内部函数
  template <typename... Indices>
  void check_indices(Indices... indices) const {
    const size_t idx_array[sizeof...(lengths)] = {static_cast<size_t>(indices)...};
    for (size_t i = 0; i < sizeof...(lengths); ++i) {
      if (idx_array[i] >= mdspan_.extent(i)) {
        throw std::out_of_range(std::string("Index ") + std::to_string(idx_array[i]) + " out of range for dimension " +
                                std::to_string(i) + " (size: " + std::to_string(mdspan_.extent(i)) + ")");
      }
    }
  }
};

#define DEFINE_MDARRAY_MATH_FUNC(name)                            \
  template <class T, class Layout, size_t... lengths>             \
  auto name(const mdarray<T, Layout, lengths...>& arr) noexcept { \
    return arr.name();                                            \
  }

// 批量定义数学函数
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

template <class T, class Layout, size_t... lengths>
auto pow(const mdarray<T, Layout, lengths...>& arr, T y) noexcept {
  return arr.pow(y);
}

template <class T, class Layout, size_t... lengths>
auto exp(const mdarray<T, Layout, lengths...>& arr, T y) noexcept {
  return arr.exp(y);
}

// 常用别名
template <class T, size_t... lengths>
using mdarray_row_major = mdarray<T, std::layout_right, lengths...>;

template <class T, size_t... lengths>
using mdarray_col_major = mdarray<T, std::layout_left, lengths...>;

template <class T, size_t N>
using array_1d = mdarray_row_major<T, N>;

template <class T, size_t N1, size_t N2>
using array_2d = mdarray_row_major<T, N1, N2>;

template <class T, size_t N1, size_t N2, size_t N3>
using array_3d = mdarray_row_major<T, N1, N2, N3>;

template <class T, size_t N1, size_t N2, size_t N3, size_t N4>
using array_4d = mdarray_row_major<T, N1, N2, N3, N4>;

template <class T, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5>
using array_5d = mdarray_row_major<T, N1, N2, N3, N4, N5>;

template <class T, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5, size_t N6>
using array_6d = mdarray_row_major<T, N1, N2, N3, N4, N5, N6>;

#endif  // __MDVECTOR_ENGINE_STATIC_H__