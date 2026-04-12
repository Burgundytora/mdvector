#ifndef __MDVECTOR_INPLACE_VECTOR__
#define __MDVECTOR_INPLACE_VECTOR__

#include "common/detail.h"
#include "common/iterator_mixin.h"
#include "common/type_concept.h"
#include "expression_template/operator_overload.h"
#include "simd/simd_function.h"
#include "common/math_function.h"

namespace md {

template <typename T, size_t Rank, size_t Capacity, typename Layout = std::layout_right>
class inplace_vector : public md::base_expr<inplace_vector<T, Rank, Capacity, Layout>, T>,
                       public md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T> {
 public:
  using Policy = md::aligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

 private:
  /// 成员变量
  static constexpr size_t capacity_ = (Capacity % md::simd<T>::pack_size == 0)
                                          ? Capacity
                                          : ((Capacity / md::simd<T>::pack_size) + 1) * md::simd<T>::pack_size;
  alignas(md::simd<T>::alignment) std::array<T, capacity_> array_;
  std::array<std::size_t, Rank> shape_;
  size_t size_;
  std::mdspan<T, std::dextents<size_t, Rank>, Layout> mdspan_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  inplace_vector() = default;

  explicit inplace_vector(const std::array<std::size_t, Rank>& shape)
      : shape_(shape),
        size_(md::calculate_size(shape)),
        mdspan_(create_mdspan(shape, std::make_index_sequence<Rank>{})) {
    check_capacity();
  }

  template <typename... Sizes>
    requires(sizeof...(Sizes) == Rank && (std::is_convertible_v<Sizes, size_t> && ...))
  explicit inplace_vector(Sizes... sizes)
      : shape_(std::array<size_t, Rank>{static_cast<size_t>(sizes)...}),
        size_(md::calculate_size(shape_)),
        mdspan_(create_mdspan(shape_, std::make_index_sequence<Rank>{})) {
    check_capacity();
  }

  ~inplace_vector() = default;

  inplace_vector(const inplace_vector& other)
      : array_(other.array_),
        shape_(other.shape_),
        size_(other.size_),
        mdspan_(create_mdspan(other.shape_, std::make_index_sequence<Rank>{})) {
    check_capacity();
  }

  inplace_vector(inplace_vector&& other) noexcept
      : array_(std::move(other.array_)),
        shape_(std::move(other.shape_)),
        size_(other.size_),
        mdspan_(std::move(other.mdspan_)) {
    check_capacity();
  }

  inplace_vector& operator=(const inplace_vector& other) {
    if (this != &other) {
      array_ = other.array_;
      shape_ = other.shape_;
      size_ = other.size_;
      mdspan_ = create_mdspan(other.shape_, std::make_index_sequence<Rank>{});
      check_capacity();
    }

    return *this;
  }

  inplace_vector& operator=(inplace_vector&& other) noexcept {
    if (this != &other) {
      array_ = std::move(other.array_);
      shape_ = std::move(other.shape_);
      size_ = other.size_;
      mdspan_ = std::move(other.mdspan_);
      check_capacity();
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
    if (linear_index >= size()) {
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
    if (linear_index >= size() || dim >= Rank) {
      throw std::out_of_range("Index out of range");
    }

    return get_md_index(linear_index)[dim];
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 访问属性
  T* data() { return array_.data(); }

  const T* data() const { return array_.data(); }

  size_t used_size() const { return capacity_; }

  size_t size() const { return size_; }

  auto extents() const { return shape_; }

  size_t extent(int index) const { return shape_.at(index); }

  bool empty() { return false; }

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
      *iterator(this, i) = current;
      current += step;
    }
  }

  void set_shape(std::array<size_t, Rank> shape) {
    if (shape == shape_ && !mdspan_.empty()) {
      return;
    }
    size_ = md::calculate_size(shape);
    check_capacity();
    shape_ = shape;
    mdspan_ = create_mdspan(shape, std::make_index_sequence<Rank>{});
  }

  template <typename... Sizes>
    requires(sizeof...(Sizes) == Rank && (std::is_convertible_v<Sizes, size_t> && ...))
  void set_shape(Sizes... sizes) {
    std::array<size_t, Rank> new_shape{static_cast<size_t>(sizes)...};
    set_shape(new_shape);
  }

  void set_random_uniform(T min_val = 0, T max_val = 1)
    requires Numeric<T>
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < size_; ++i) {
        array_[i] = dis(gen);
      }
    } else {
      std::uniform_int_distribution<T> dis(min_val, max_val);
      for (size_t i = 0; i < size_; ++i) {
        array_[i] = dis(gen);
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
      array_[i] = dis(gen);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 打印
  void print() const
    requires Printable<T>
  {
    md::print_mdspan(mdspan_);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 迭代器
  using md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T>::begin;
  using md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T>::end;
  using md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T>::cbegin;
  using md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T>::cend;
  using md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T>::rbegin;
  using md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T>::rend;
  using md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T>::crbegin;
  using md::iterator_mixin<inplace_vector<T, Rank, Capacity, Layout>, T>::crend;

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 表达式模板数值计算
  template <typename E>
  inplace_vector& operator=(const md::base_expr<E, T>& expr) noexcept
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
  void store_simd(const size_t& i, md::simd<T2>::const_ref_type simd_val) noexcept {
    return Policy::store<T>(this->data() + i, simd_val);
  }

  template <typename T2>
  void store_simd_mask(const size_t& i, const size_t& remaining, md::simd<T2>::const_ref_type simd_val) noexcept {
    return Policy::mask_store<T>(this->data() + i, remaining, simd_val);
  }

  inplace_vector& operator+=(const inplace_vector& other) noexcept
    requires Numeric<T>
  {
    md::simd_add_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  inplace_vector& operator-=(const inplace_vector& other) noexcept
    requires Numeric<T>
  {
    md::simd_sub_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  inplace_vector& operator*=(const inplace_vector& other) noexcept
    requires Numeric<T>
  {
    md::simd_mul_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  inplace_vector& operator/=(const inplace_vector& other) noexcept
    requires Numeric<T>
  {
    md::simd_div_inplace<T, Policy>(this->data(), other.data(), this->used_size());
    return *this;
  }

  template <typename E>
  inplace_vector& operator+=(const md::base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this + expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  inplace_vector& operator-=(const md::base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this - expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  inplace_vector& operator*=(const md::base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this * expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  inplace_vector& operator/=(const md::base_expr<E, T>& expr) noexcept
    requires Numeric<T>
  {
    (*this / expr).template eval_to<>(*this);
    return *this;
  }

  inplace_vector& operator+=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_add_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  inplace_vector& operator-=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_sub_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  inplace_vector& operator*=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_mul_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  inplace_vector& operator/=(T scalar) noexcept
    requires Numeric<T>
  {
    md::simd_div_inplace_scalar<T, Policy>(this->data(), scalar, this->used_size());
    return *this;
  }

  // 取负
  auto operator-() const noexcept
    requires Numeric<T>
  {
    md::inplace_vector<T, Rank, Capacity, Layout> result(this->extents());
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
  template <size_t... Indices>
  auto create_mdspan(const std::array<size_t, Rank>& shape, std::index_sequence<Indices...>) {
    return std::mdspan<T, std::dextents<size_t, Rank>, Layout>(data(), shape[Indices]...);
  }

  void check_initialized() const {
    if (mdspan_.empty()) {
      throw std::logic_error("md::inplace_vector not initialized");
    }
  }

  void check_capacity() {
    if (size_ > capacity_) {
      std::println("md::inplace_vector size({}) larger than capacity({})", size_, capacity_);
      throw std::out_of_range("md::inplace_vector size larger than capacity");
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
  std::size_t calculate_offset(const std::array<md::slice, Rank>& slices, const std::array<bool, Rank>& is_integral) {
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

}  // namespace md

#endif  // __MDVECTOR_INPLACE_VECTOR__