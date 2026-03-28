#ifndef __MDVECTOR_SPAN__
#define __MDVECTOR_SPAN__

#include "common/detail.h"
#include "common/iterator_mixin.h"
#include "common/type_concept.h"
#include "expression_template/operator_overload.h"
#include "simd/simd_function.h"
#include "common/math_function.h"

namespace md {

template <typename T, size_t Rank, typename Layout = std::layout_right>
class span : public base_expr<span<T, Rank, Layout>, T>, public iterator_mixin<span<T, Rank, Layout>, T> {
 public:
  using Policy = unaligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

 protected:
  std::mdspan<T, std::dextents<size_t, Rank>, Layout> mdspan_;
  std::array<size_t, Rank> shape_;
  size_t size_;
  size_t align_size_;
  size_t remaining_size_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  constexpr span() noexcept = default;

  span(T* data, const std::array<std::size_t, Rank>& shape)
      : mdspan_(create_mdspan(data, shape, std::make_index_sequence<Rank>{})),
        shape_(shape),
        size_(calculate_size(shape)),
        align_size_(get_aligned_size<T>(size_)),
        remaining_size_(size_ > simd<T>::pack_size ? align_size_ - size_ : size_) {}

  span(const span& other) = delete;

  span(const span&& other) = delete;

  span& operator=(const span& other) = delete;

  // 删除移动赋值 不管理所有权
  span& operator=(span&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~span() = default;

  template <typename E>
  span(const base_expr<E, T>& expr) = delete;

  template <typename E>
  span& operator=(const base_expr<E, T>& expr) noexcept {
    expr.template eval_to<>(*this);
    return *this;
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 访问属性
  T* data() { return mdspan_.data_handle(); }

  const T* data() const { return mdspan_.data_handle(); }

  size_t used_size() const noexcept { return align_size_; }

  size_t size() const noexcept { return size_; }

  auto extents() const { return shape_; }

  size_t extent(int index) const { return shape_.at(index); }

  bool empty() { return mdspan_.empty(); }

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
  /// simd接口
  template <typename T2>
  typename simd<T2>::type load_simd(size_t i) const noexcept {
    if (i + simd<T2>::pack_size <= size_) {
      return Policy::load<T2>(this->data() + i);
    } else {
      return Policy::mask_load<T2>(this->data() + i, remaining_size_);
    }
  }

  template <typename T2>
  void store_simd(const size_t& i, simd<T2>::const_ref_type simd_val) noexcept {
    if (i + simd<T2>::pack_size <= size_) {
      Policy::store<T>(this->data() + i, simd_val);
    } else {
      Policy::mask_store<T>(this->data() + i, remaining_size_, simd_val);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 表达式模板数值计算
  template <typename E>
  span& operator+=(const base_expr<E, T>& expr) noexcept {
    (*this + expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  span& operator-=(const base_expr<E, T>& expr) noexcept {
    (*this - expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  span& operator*=(const base_expr<E, T>& expr) noexcept {
    (*this * expr).template eval_to<>(*this);
    return *this;
  }

  template <typename E>
  span& operator/=(const base_expr<E, T>& expr) noexcept {
    (*this / expr).template eval_to<>(*this);
    return *this;
  }

  span& operator+=(T scalar) noexcept {
    (*this + scalar).template eval_to<>(*this);
    return *this;
  }

  span& operator-=(T scalar) noexcept {
    (*this - scalar).template eval_to<>(*this);
    return *this;
  }

  span& operator*=(T scalar) noexcept {
    (*this * scalar).template eval_to<>(*this);
    return *this;
  }

  span& operator/=(T scalar) noexcept {
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

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 打印
  void print() const
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      print_mdspan(mdspan_);
    } else {
      throw std::logic_error("md::span need to be initialized before print!!!");
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 迭代器
  using iterator_mixin<span<T, Rank, Layout>, T>::begin;
  using iterator_mixin<span<T, Rank, Layout>, T>::end;
  using iterator_mixin<span<T, Rank, Layout>, T>::cbegin;
  using iterator_mixin<span<T, Rank, Layout>, T>::cend;
  using iterator_mixin<span<T, Rank, Layout>, T>::rbegin;
  using iterator_mixin<span<T, Rank, Layout>, T>::rend;
  using iterator_mixin<span<T, Rank, Layout>, T>::crbegin;
  using iterator_mixin<span<T, Rank, Layout>, T>::crend;

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

#endif  // __MDVECTOR_SPAN__