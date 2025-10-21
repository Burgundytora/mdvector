#ifndef __MDVECTOR_H__
#define __MDVECTOR_H__

#include <vector>

#include "common/detail.h"
#include "common/type_concept.h"
#include "expression_template/operator.h"
#include "simd/allocator.h"
#include "simd/simd_function.h"
#include "span.h"

template <class T, size_t Rank, class Layout>
class mdvector : public md::tensor_expr<mdvector<T, Rank, Layout>, T> {
  /// simd对齐策略
  using Policy = md::aligned_policy;

 private:
  /// 成员变量
  std::vector<T, md::auto_allocator<T>> vector_;
  std::array<size_t, Rank> shape_;
  size_t size_;
  std::mdspan<T, std::dextents<size_t, Rank>, Layout> mdspan_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  mdvector() = default;

  explicit mdvector(const std::array<std::size_t, Rank>& shape)
      : vector_(md::calculate_size(shape)),
        shape_(shape),
        size_(md::calculate_size(shape)),
        mdspan_(create_mdspan(shape, std::make_index_sequence<Rank>{})) {}

  ~mdvector() = default;

  mdvector(const mdvector& other)
      : vector_(other.vector_),
        shape_(other.shape_),
        size_(other.size_),
        mdspan_(create_mdspan(other.shape_, std::make_index_sequence<Rank>{})) {}

  mdvector(mdvector&& other) noexcept
      : vector_(std::move(other.vector_)),
        shape_(std::move(other.shape_)),
        size_(other.size_),
        mdspan_(std::move(other.mdspan_)) {}

  mdvector& operator=(const mdvector& other) {
    if (this != &other) {
      vector_ = other.vector_;
      shape_ = other.shape_;
      size_ = other.size_;
      mdspan_ = create_mdspan(other.shape_, std::make_index_sequence<Rank>{});
    }
    return *this;
  }

  mdvector& operator=(mdvector&& other) noexcept {
    if (this != &other) {
      vector_ = std::move(other.vector_);
      shape_ = std::move(other.shape_);
      size_ = other.size_;
      mdspan_ = std::move(other.mdspan_);
    }
    return *this;
  }

  // 从span创建
  mdvector(const md::span<T, Rank, Layout>& span) noexcept
    requires Numeric<T>
  {
    this->set_shape(span.extents());
    span.template eval_to<T, Policy>(this->data());
  }

  mdvector& operator=(const md::span<T, Rank, Layout>& span) noexcept
    requires Numeric<T>
  {
    span.template eval_to<T, Policy>(this->data());
    return *this;
  }

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
    check_initialized();
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <class... Indices>
  const T& at(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    check_initialized();
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <class... Indices>
  size_t get_1d_index(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    // 使用 mdspan 的 mapping 来获取线性索引
    return mdspan_.mapping()(indices...);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 访问属性
  T* data() { return vector_.data(); }

  const T* data() const { return vector_.data(); }

  size_t used_size() const { return size_; }

  size_t size() const { return size_; }

  size_t extent(int index) const { return shape_.at(index); }

  auto extents() const { return shape_; }

  constexpr size_t rank() const { return Rank; }

  bool empty() { return vector_.empty(); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 更改属性
  void fill(T val) { std::fill(begin(), end(), val); }

  void set_shape(std::array<size_t, Rank> shape) {
    if (shape == shape_ && !mdspan_.empty()) {
      return;
    }
    size_ = md::calculate_size(shape);
    vector_.resize(size_);
    shape_ = shape;
    mdspan_ = create_mdspan(shape, std::make_index_sequence<Rank>{});
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 打印
  void print()
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      md::print_mdspan(mdspan_);
    } else {
      throw std::logic_error("mdvector need to be initialized before print!!!");
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 创建span 内存连续视图
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
      return data() + offset;
    } else {
      return md::span<T, NewRank, Layout>(data() + offset, new_extents);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 迭代器
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() noexcept { return vector_.data(); }
  iterator end() noexcept { return vector_.data() + vector_.size(); }
  const_iterator begin() const noexcept { return vector_.data(); }
  const_iterator end() const noexcept { return vector_.data() + vector_.size(); }
  const_iterator cbegin() const noexcept { return vector_.data(); }
  const_iterator cend() const noexcept { return vector_.data() + vector_.size(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 表达式模板数值计算
  template <class E>
  mdvector(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    this->set_shape(expr.extents());
    expr.template eval_to<T, Policy>(this->data());
  }

  template <class E>
  mdvector& operator=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    this->set_shape(expr.extents());
    expr.template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd(size_t i) const noexcept
    requires Numeric<T>
  {
    return md::simd<T2>::load(data() + i);
  }

  template <class T2>
  typename md::simd<T2>::type eval_simd_mask(size_t i) const noexcept
    requires Numeric<T>
  {
    return md::simd<T2>::mask_load(data() + i, used_size() - i);
  }

  mdvector& operator+=(const mdvector& other) noexcept
    requires Numeric<T>
  {
    md::simd_add_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdvector& operator-=(const mdvector& other) noexcept
    requires Numeric<T>
  {
    md::simd_sub_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdvector& operator*=(const mdvector& other) noexcept
    requires Numeric<T>
  {
    md::simd_mul_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  mdvector& operator/=(const mdvector& other) noexcept
    requires Numeric<T>
  {
    md::simd_div_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  template <class E>
  mdvector& operator+=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this + expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator-=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this - expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator*=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this * expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  template <class E>
  mdvector& operator/=(const md::tensor_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this / expr).template eval_to<T, Policy>(this->data());
    return *this;
  }

  mdvector& operator+=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_add_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdvector& operator-=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_sub_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdvector& operator*=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_mul_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  mdvector& operator/=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_div_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 数学函数
  // 求和
  T sum() { return std::reduce(begin(), end()); }

  // 求积
  T prod() { return std::reduce(begin(), end(), T(1), std::multiplies<T>()); }

  // 最大值
  T max() { return *std::max_element(begin(), end()); }

  // 最小值
  T min() { return *std::min_element(begin(), end()); }

  // 平均值
  T mean() { return std::reduce(begin(), end()) / T(size()); }

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
    auto vec = vector_;
    std::sort(vec.begin(), vec.end());
    size_t size = vec.size();
    if (size % 2 == 0) {
      return (static_cast<T>(vec[size / 2 - 1]) + static_cast<T>(vec[size / 2])) / 2.0;
    } else {
      return static_cast<T>(vec[size / 2]);
    }
  }

  // 数学函数简化定义
  using this_type = mdvector;
#define DEFINE_MD_MATH_OP(name, op)                                                                       \
  this_type name() const noexcept                                                                         \
    requires Numeric<T>                                                                                   \
  {                                                                                                       \
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

  this_type exp(T y) const noexcept
    requires Numeric<T>
  {
    this_type res(*this);
    std::transform(this->begin(), this->end(), res.begin(), [y](T val) noexcept { return std::pow(y, val); });
    return res;
  }

  this_type pow(T y) const noexcept
    requires Numeric<T>
  {
    this_type res(*this);
    std::transform(this->begin(), this->end(), res.begin(), [y](T val) noexcept { return std::pow(val, y); });
    return res;
  }

 private:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 内部函数
  template <size_t... Indices>
  auto create_mdspan(const std::array<size_t, Rank>& shape, std::index_sequence<Indices...>) {
    return std::mdspan<T, std::dextents<size_t, Rank>, Layout>(vector_.data(), shape[Indices]...);
  }

  void check_initialized() const {
    if (mdspan_.empty()) {
      throw std::logic_error("mdvector not initialized");
    }
  }

  template <typename... Indices>
  void check_indices(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match the rank of mdvector");

    const size_t idx_array[Rank] = {static_cast<size_t>(indices)...};
    for (size_t i = 0; i < Rank; ++i) {
      if (idx_array[i] >= shape_[i]) {
        throw std::out_of_range(
            std::format("Index {} out of range for dimension {} (size: {})", idx_array[i], i, shape_[i]));
      }
    }
  }

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
///////////////////////////////////////////////////////////////////////////////////////

// 视图的数学函数返回一个新的mdvector
#define DEFINE_SPAN_MATH_FUNC(name, func)                                                                \
  template <class T, size_t Rank, class Layout>                                                          \
  mdvector<T, Rank, Layout> md::span<T, Rank, Layout>::name() const noexcept                             \
    requires Numeric<T>                                                                                  \
  {                                                                                                      \
    mdvector<T, Rank, Layout> res(this->extents());                                                      \
    std::transform(this->begin(), this->end(), res.begin(), [](T val) noexcept { return (func)(val); }); \
    return res;                                                                                          \
  }

DEFINE_SPAN_MATH_FUNC(cos, std::cos);
DEFINE_SPAN_MATH_FUNC(acos, std::acos);
DEFINE_SPAN_MATH_FUNC(cosh, std::cosh);
DEFINE_SPAN_MATH_FUNC(sin, std::sin);
DEFINE_SPAN_MATH_FUNC(asin, std::asin);
DEFINE_SPAN_MATH_FUNC(sinh, std::sinh);
DEFINE_SPAN_MATH_FUNC(tan, std::tan);
DEFINE_SPAN_MATH_FUNC(atan, std::atan);
DEFINE_SPAN_MATH_FUNC(tanh, std::tanh);
DEFINE_SPAN_MATH_FUNC(abs, std::abs);
DEFINE_SPAN_MATH_FUNC(sqrt, std::sqrt);
DEFINE_SPAN_MATH_FUNC(log10, std::log10);
DEFINE_SPAN_MATH_FUNC(ln, std::log);

#undef DEFINE_SPAN_MATH_FUNC

template <class T, size_t Rank, class Layout>
mdvector<T, Rank, Layout> md::span<T, Rank, Layout>::exp(T y) const noexcept
  requires Numeric<T>
{
  mdvector<T, Rank, Layout> res(this->extents());
  std::transform(this->begin(), this->end(), res.data_.begin(), [y](T val) noexcept { return std::pow(y, val); });
  return res;
}

template <class T, size_t Rank, class Layout>
mdvector<T, Rank, Layout> md::span<T, Rank, Layout>::pow(T y) const noexcept
  requires Numeric<T>
{
  mdvector<T, Rank, Layout> res(this->extents());
  std::transform(this->begin(), this->end(), res.begin(), [y](T val) noexcept { return std::pow(val, y); });
  return res;
}

///////////////////////////////////////////////////////////////////////////////////////
/// mdvector类外数学函数
#define DEFINE_MD_MATH_FUNC(name)                          \
  template <class T, size_t Rank, class Layout>            \
  auto name(const mdvector<T, Rank, Layout>& v) noexcept   \
    requires Numeric<T>                                    \
  {                                                        \
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

// 从表达式创建mdvector 数学表达式的临时变量
#define DEFINE_EXPR_MATH_FUNC(name, func)                                                                 \
  template <class T, class E>                                                                             \
  mdvector<T, 1> name(const md::tensor_expr<E, T>& expr) noexcept                                         \
    requires Numeric<T>                                                                                   \
  {                                                                                                       \
    mdvector<T, 1> res = expr;                                                                            \
    std::transform(res.begin(), res.end(), res.begin(), [](double val) noexcept { return (func)(val); }); \
    return res;                                                                                           \
  }

DEFINE_EXPR_MATH_FUNC(cos, std::cos)
DEFINE_EXPR_MATH_FUNC(acos, std::acos)
DEFINE_EXPR_MATH_FUNC(cosh, std::cosh)
DEFINE_EXPR_MATH_FUNC(sin, std::sin)
DEFINE_EXPR_MATH_FUNC(asin, std::asin)
DEFINE_EXPR_MATH_FUNC(sinh, std::sinh)
DEFINE_EXPR_MATH_FUNC(tan, std::tan)
DEFINE_EXPR_MATH_FUNC(atan, std::atan)
DEFINE_EXPR_MATH_FUNC(tanh, std::tanh)
DEFINE_EXPR_MATH_FUNC(abs, std::abs)
DEFINE_EXPR_MATH_FUNC(sqrt, std::sqrt)
DEFINE_EXPR_MATH_FUNC(log10, std::log10)
DEFINE_EXPR_MATH_FUNC(ln, std::log)

#undef DEFINE_EXPR_MATH_FUNC

template <class T, class E>
mdvector<T, 1> exp(const md::tensor_expr<E, T>& expr, T y) noexcept
  requires Numeric<T>
{
  mdvector<T, 1> res = expr;
  std::transform(res.begin(), res.end(), res.begin(), [y](T val) noexcept { return std::pow(y, val); });
  return res;
}

template <class T, class E>
mdvector<T, 1> pow(const md::tensor_expr<E, T>& expr, T y) noexcept
  requires Numeric<T>
{
  mdvector<T, 1> res = expr;
  std::transform(res.begin(), res.end(), res.begin(), [y](T val) noexcept { return std::pow(val, y); });
  return res;
}

// 常用别名
template <typename T, size_t Rank>
using mdvector_row_major = mdvector<T, Rank, std::layout_right>;

template <typename T, size_t Rank>
using mdvector_col_major = mdvector<T, Rank, std::layout_left>;

template <size_t Rank>
using mdshape = std::array<size_t, Rank>;

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