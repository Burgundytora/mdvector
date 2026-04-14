#ifndef __MDVECTOR_MDVECTOR__
#define __MDVECTOR_MDVECTOR__

#include "common/detail.h"
#include "common/iterator_mixin.h"
#include "common/base_concept.h"
#include "expression/operator_overload.h"
#include "simd/allocator.h"
#include "simd/simd_function.h"
#include "common/math_function.h"
#include "span.h"
#include "view.h"

namespace md {

template <typename T, size_t Rank, typename Layout>
class vector : public base_expr<vector<T, Rank, Layout>, T>, public iterator_mixin<vector<T, Rank, Layout>, T> {
 public:
  using Policy = aligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

 private:
  /// 成员变量
  std::array<size_t, Rank> shape_;
  size_t size_;
  std::vector<T, auto_allocator<T>> vector_;
  std::mdspan<T, std::dextents<size_t, Rank>, Layout> mdspan_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  vector() = default;

  explicit vector(const std::array<std::size_t, Rank>& shape)
      : shape_(shape),
        size_(calculate_size(shape)),
        vector_(get_aligned_size<T>(size_)),
        mdspan_(create_mdspan(shape, std::make_index_sequence<Rank>{})) {}

  template <typename... Sizes>
    requires(sizeof...(Sizes) == Rank && (std::is_convertible_v<Sizes, size_t> && ...))
  explicit vector(Sizes... sizes)
      : shape_(std::array<size_t, Rank>{static_cast<size_t>(sizes)...}),
        size_(calculate_size(shape_)),
        vector_(get_aligned_size<T>(size_)),
        mdspan_(create_mdspan(shape_, std::make_index_sequence<Rank>{})) {}

  ~vector() = default;

  vector(const vector& other)
      : shape_(other.shape_),
        size_(other.size_),
        vector_(other.vector_),
        mdspan_(create_mdspan(other.shape_, std::make_index_sequence<Rank>{})) {}

  vector(vector&& other) noexcept
      : shape_(std::move(other.shape_)),
        size_(other.size_),
        vector_(std::move(other.vector_)),
        mdspan_(std::move(other.mdspan_)) {}

  vector& operator=(const vector& other) {
    if (this != &other) {
      shape_ = other.shape_;
      size_ = other.size_;
      vector_ = other.vector_;
      mdspan_ = create_mdspan(other.shape_, std::make_index_sequence<Rank>{});
    }
    return *this;
  }

  vector& operator=(vector&& other) noexcept {
    if (this != &other) {
      shape_ = std::move(other.shape_);
      size_ = other.size_;
      vector_ = std::move(other.vector_);
      mdspan_ = std::move(other.mdspan_);
    }
    return *this;
  }

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
    check_initialized();
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& at(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    check_initialized();
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
    } else {
      // 列优先布局 (Fortran-style)
      size_t remaining = linear_index;
      for (int i = 0; i < Rank; ++i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
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
  /// 访问属性
  T* data() { return vector_.data(); }

  const T* data() const { return vector_.data(); }

  size_t used_size() const { return vector_.size(); }

  size_t size() const { return size_; }

  size_t extent(int index) const { return shape_.at(index); }

  auto extents() const { return shape_; }

  constexpr size_t rank() const { return Rank; }

  bool empty() { return vector_.empty(); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 更改属性
  void fill(T val) { std::fill(begin(), end(), val); }

  void set_zeros()
    requires Numeric<T>
  {
    fill(static_cast<T>(0));
  }

  void set_ones()
    requires Numeric<T>
  {
    fill(static_cast<T>(1));
  }

  void set_arange(T start = 0, T step = 1)
    requires Numeric<T>
  {
    T current = start;
    for (size_t i = 0; i < size_; ++i) {
      vector_[i] = current;
      current += step;
    }
  }

  void set_random_uniform(T min_val = 0, T max_val = 1)
    requires Numeric<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < size_; ++i) {
        vector_[i] = dis(gen);
      }
    } else {
      std::uniform_int_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < size_; ++i) {
        vector_[i] = dis(gen);
      }
    }
  }

  void set_random_normal(T mean = 0, T stddev = 1)
    requires Numeric<T> && std::is_floating_point_v<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dis(mean, stddev);
    for (size_t i = 0; i < size_; ++i) {
      vector_[i] = dis(gen);
    }
  }

  void set_shape(std::array<size_t, Rank> shape) {
    if (shape == shape_ && !mdspan_.empty()) {
      return;
    }
    size_ = calculate_size(shape);
    vector_.resize(get_aligned_size<T>(size_));
    shape_ = shape;
    mdspan_ = create_mdspan(shape, std::make_index_sequence<Rank>{});
  }

  template <typename... Sizes>
    requires(sizeof...(Sizes) == Rank && (std::is_convertible_v<Sizes, size_t> && ...))
  void set_shape(Sizes... sizes) {
    std::array<size_t, Rank> new_shape{static_cast<size_t>(sizes)...};
    set_shape(new_shape);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 打印
  void print() const
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      print_mdspan(mdspan_);
    } else {
      throw std::logic_error("vector need to be initialized before print!!!");
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 创建span 内存连续视图
  template <typename... Slices>
  auto span(Slices... slices) {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");

    constexpr std::size_t NewRank = compressed_rank_v<Slices...>;

    auto [slice_array, is_integral] = prepare_slices<Rank>(extents(), slices...);

    // 检查越界
    check_slice_bounds<Rank>(slice_array, extents());

    // 检查内存连续
    if (!check_slice_contiguous<Rank, Layout>(extents(), slice_array, is_integral)) {
      throw std::runtime_error("span slices must result in contiguous memory");
    }

    // 计算新的extents
    std::array<std::size_t, NewRank> new_extents;
    std::size_t new_idx = 0;

    for (int i = 0; i < Rank; ++i) {
      if (!is_integral[i]) {  // 只保留非整数索引的维度
        const auto& s = slice_array[i];
        std::ptrdiff_t start = normalize_index(s.start, extent(i));
        std::ptrdiff_t end = normalize_index(s.end, extent(i));
        new_extents[new_idx++] = s.is_all ? extent(i) : (end - start + 1);
        if (s.step != 1) {
          throw std::invalid_argument("span slice's step must be 1.");
        }
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

  /// 创建view 视图
  template <typename... Slices>
  auto view(Slices... slices) {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");

    constexpr std::size_t NewRank = compressed_rank_v<Slices...>;

    auto [slice_array, is_integral] = prepare_slices<Rank>(extents(), slices...);

    // 检查越界
    check_slice_bounds<Rank>(slice_array, extents());

    // 计算新的extents
    std::array<std::size_t, NewRank> new_extents;
    std::size_t new_idx = 0;

    for (int i = 0; i < Rank; ++i) {
      if (!is_integral[i]) {  // 只保留非整数索引的维度
        const auto& s = slice_array[i];
        std::ptrdiff_t start = normalize_index(s.start, extent(i));
        std::ptrdiff_t end = normalize_index(s.end, extent(i));
        new_extents[new_idx++] = s.is_all ? extent(i) : 1 + std::floor((end - start) / s.step);
      }
    }

    // 计算步长
    std::array<size_t, NewRank> stride;
    new_idx = 0;
    size_t stride_single = 1;
    size_t last_extent = 1;
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        stride_single = slice_array[i].step * last_extent;
        if (!is_integral[i]) {  // 只保留非整数索引的维度
          stride[NewRank - new_idx++ - 1] = stride_single;
        }
        last_extent *= extent(i);
      }
    } else {
      for (int i = 0; i >= Rank - 1; ++i) {
        stride_single = slice_array[i].step * last_extent;
        if (!is_integral[i]) {  // 只保留非整数索引的维度
          stride[NewRank - new_idx++ - 1] = stride_single;
        }
        last_extent *= extent(i);
      }
    }

    // 计算新的数据指针偏移
    std::size_t offset = calculate_offset(slice_array, is_integral);

    // 返回适当维度的span
    if constexpr (NewRank == 0) {
      // 所有维度都是整数索引，返回标量引用
      return data() + offset;
    } else {
      return md::view<T, NewRank>(data() + offset, new_extents, stride);
    }
  }

  // ///////////////////////////////////////////////////////////////////////////////////////
  // /// 迭代器
  using iterator_mixin<vector<T, Rank, Layout>, T>::begin;
  using iterator_mixin<vector<T, Rank, Layout>, T>::end;
  using iterator_mixin<vector<T, Rank, Layout>, T>::cbegin;
  using iterator_mixin<vector<T, Rank, Layout>, T>::cend;
  using iterator_mixin<vector<T, Rank, Layout>, T>::rbegin;
  using iterator_mixin<vector<T, Rank, Layout>, T>::rend;
  using iterator_mixin<vector<T, Rank, Layout>, T>::crbegin;
  using iterator_mixin<vector<T, Rank, Layout>, T>::crend;

  ///////////////////////////////////////////////////////////////////////////////////////
  /// simd接口
  template <typename T2>
  typename simd<T2>::type load_simd(const size_t& i) const noexcept
    requires Numeric<T>
  {
    return Policy::load<T2>(data() + i);
  }

  template <typename T2>
  void store_simd(const size_t& i, simd<T2>::const_ref_type simd_val) noexcept {
    return Policy::store<T>(this->data() + i, simd_val);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 表达式模板数值计算
  template <typename E>
  vector(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    this->set_shape(expr.extents());
    expr.template eval_to<>(*this);
  }

  template <typename E>
  vector& operator=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    expr.template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  vector& operator+=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this + expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  vector& operator-=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this - expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  vector& operator*=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this * expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  vector& operator/=(const base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this / expr).template eval_to<>(*this);
    return *this;
  }

  vector& operator+=(T scalar) noexcept
    requires Numeric<T>
  {
    (*this + scalar).template eval_to<>(*this);
    return *this;
  }

  vector& operator-=(T scalar) noexcept
    requires Numeric<T>
  {
    (*this - scalar).template eval_to<>(*this);
    return *this;
  }

  vector& operator*=(T scalar) noexcept
    requires Numeric<T>
  {
    (*this * scalar).template eval_to<>(*this);
    return *this;
  }

  vector& operator/=(T scalar) noexcept
    requires Numeric<T>
  {
    (*this / scalar).template eval_to<>(*this);
    return *this;
  }

  // 取负
  auto operator-() const noexcept
    requires Numeric<T>
  {
    return (*this * static_cast<T>(-1));
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
  template <size_t... Indices>
  auto create_mdspan(const std::array<size_t, Rank>& shape, std::index_sequence<Indices...>) {
    return std::mdspan<T, std::dextents<size_t, Rank>, Layout>(vector_.data(), shape[Indices]...);
  }

  void check_initialized() const {
    if (mdspan_.empty()) {
      throw std::logic_error("md::vector not initialized");
    }
  }

  template <typename... Indices>
  void check_indices(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match the rank of md::vector");

    const size_t idx_array[Rank] = {static_cast<size_t>(indices)...};
    for (int i = 0; i < Rank; ++i) {
      if (idx_array[i] >= shape_[i]) {
        throw std::out_of_range(
            std::format("Index {} out of range for dimension {} (size: {})", idx_array[i], i, shape_[i]));
      }
    }
  }

  // 计算数据指针偏移
  std::size_t calculate_offset(const std::array<slice, Rank>& slices, const std::array<bool, Rank>& is_integral) {
    std::size_t offset = 0;
    std::size_t stride = 1;

    // 按内存布局计算偏移（这里以行优先为例）
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        if (!is_integral[i]) {
          offset += slices[i].start * stride;
          stride *= extent(i);
        } else {
          offset += static_cast<std::size_t>(slices[i].start) * stride;
        }
      }
    } else {
      for (int i = 0; i <= Rank - 1; ++i) {
        if (!is_integral[i]) {
          offset += slices[i].start * stride;
          stride *= extent(i);
        } else {
          offset += static_cast<std::size_t>(slices[i].start) * stride;
        }
      }
    }

    return offset;
  }
};

template <size_t Rank>
using shape = std::array<size_t, Rank>;

}  // namespace md

// 常用别名
template <typename T, size_t Rank>
using mdvector_row_major = md::vector<T, Rank, std::layout_right>;

template <typename T, size_t Rank>
using mdvector_col_major = md::vector<T, Rank, std::layout_left>;

using shape_1d = std::array<size_t, 1>;
using shape_2d = std::array<size_t, 2>;
using shape_3d = std::array<size_t, 3>;
using shape_4d = std::array<size_t, 4>;
using shape_5d = std::array<size_t, 5>;
using shape_6d = std::array<size_t, 6>;

template <typename T>
using vector_1d = md::vector<T, 1>;

template <typename T>
using vector_2d = md::vector<T, 2>;

template <typename T>
using vector_3d = md::vector<T, 3>;

template <typename T>
using vector_4d = md::vector<T, 4>;

template <typename T>
using vector_5d = md::vector<T, 5>;

template <typename T>
using vector_6d = md::vector<T, 6>;

#endif  // __MDVECTOR_MDVECTOR__