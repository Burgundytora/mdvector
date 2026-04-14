// core/multi_dim.h
#ifndef __MDVECTOR_CORE_MULTI_DIM__
#define __MDVECTOR_CORE_MULTI_DIM__

#include "../common/mdspan_little.h"
#include <format>

namespace md {

// ============================================================================
// mdspan_traits - 主模板
// ============================================================================
template <typename T, size_t Rank, typename Layout, typename Enable = void>
struct mdspan_traits {
  using type = std::mdspan<T, std::dextents<size_t, Rank>, Layout>;

  static auto create(T* data, const std::array<size_t, Rank>& shape) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
      return type(data, shape[Is]...);
    }(std::make_index_sequence<Rank>{});
  }
};

// ============================================================================
// 静态维度特化 - 使用完全不同的模板参数签名
// 注意：偏特化的模板参数必须与主模板有相同的"结构"
// ============================================================================
template <typename T, size_t... Lengths>
struct static_mdspan_helper;

template <typename T, size_t... Lengths>
struct static_mdspan_helper {
  static constexpr size_t Rank = sizeof...(Lengths);
  using type = std::mdspan<T, std::extents<size_t, Lengths...>>;
  static constexpr std::array<size_t, Rank> shape = {Lengths...};

  static auto create(T* data) { return type(data); }
};

// ============================================================================
// multi_dim_impl - 主模板
// ============================================================================
template <typename Derived, typename T, size_t Rank, typename Layout>
class multi_dim_impl {
 protected:
  std::array<size_t, Rank> shape_;
  using MdspanType = typename mdspan_traits<T, Rank, Layout>::type;
  MdspanType mdspan_;

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

 public:
  multi_dim_impl() = default;

  void init_mdspan() { mdspan_ = mdspan_traits<T, Rank, Layout>::create(derived().data(), shape_); }

  auto extents() const noexcept { return shape_; }
  size_t extent(size_t dim) const noexcept { return shape_[dim]; }

  void set_shape(const std::array<size_t, Rank>& shape) {
    shape_ = shape;
    init_mdspan();
  }

  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank);
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank);
    return mdspan_[indices...];
  }

  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == Rank);
    return operator()(indices...);
  }

  template <typename... Indices>
  const T& operator[](Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank);
    return operator()(indices...);
  }

  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank);
    check_indices(indices...);
    return operator()(indices...);
  }

  template <typename... Indices>
  const T& at(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank);
    check_indices(indices...);
    return operator()(indices...);
  }

  template <typename... Indices>
  size_t get_1d_index(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank);
    return mdspan_.mapping()(indices...);
  }

  std::array<size_t, Rank> get_md_index(size_t linear_index) const {
    if (linear_index >= derived().size()) {
      throw std::out_of_range("Linear index out of range");
    }
    std::array<size_t, Rank> indices{};
    size_t remaining = linear_index;
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
      }
    } else if constexpr (std::is_same_v<Layout, std::layout_left>) {
      for (size_t i = 0; i < Rank; ++i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
      }
    } else {
      for (size_t i = 0; i < Rank; ++i) {
        indices[i] = (linear_index / mdspan_.stride(i)) % shape_[i];
      }
    }
    return indices;
  }

  size_t get_dim_index(size_t linear_index, size_t dim) const {
    if (linear_index >= derived().size() || dim >= Rank) {
      throw std::out_of_range("Index out of range");
    }
    return get_md_index(linear_index)[dim];
  }

  void print() const
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      print_mdspan(mdspan_);
    } else {
      throw std::logic_error("Object not initialized");
    }
  }

  const MdspanType& mdspan() const { return mdspan_; }
  MdspanType& mdspan() { return mdspan_; }

 protected:
  template <typename... Indices>
  void check_indices(Indices... indices) const {
    const size_t idx_array[Rank] = {static_cast<size_t>(indices)...};
    for (size_t i = 0; i < Rank; ++i) {
      if (idx_array[i] >= shape_[i]) {
        throw std::out_of_range(std::format("Index {} out of range (dim {}, size {})", idx_array[i], i, shape_[i]));
      }
    }
  }
};

// ============================================================================
// view 偏特化 (layout_stride)
// ============================================================================
template <typename Derived, typename T, size_t Rank>
class multi_dim_impl<Derived, T, Rank, std::layout_stride> {
 protected:
  std::array<size_t, Rank> shape_;
  std::array<size_t, Rank> stride_;
  using MdspanType = std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride>;
  MdspanType mdspan_;

  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

 public:
  multi_dim_impl() = default;

  auto extents() const noexcept { return shape_; }
  size_t extent(size_t dim) const noexcept { return shape_[dim]; }
  auto strides() const noexcept { return stride_; }
  size_t stride(size_t dim) const noexcept { return stride_[dim]; }

  void init_mdspan(T* data, const std::array<size_t, Rank>& shape, const std::array<size_t, Rank>& stride) {
    shape_ = shape;
    stride_ = stride;
    mdspan_ = MdspanType(data, std::layout_stride::mapping(std::dextents<size_t, Rank>(shape), stride));
  }

  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank);
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank);
    return mdspan_[indices...];
  }

  template <typename... Indices>
  T& operator[](Indices... indices) {
    return operator()(indices...);
  }

  template <typename... Indices>
  const T& operator[](Indices... indices) const {
    return operator()(indices...);
  }

  template <typename... Indices>
  T& at(Indices... indices) {
    check_indices(indices...);
    return operator()(indices...);
  }

  template <typename... Indices>
  const T& at(Indices... indices) const {
    check_indices(indices...);
    return operator()(indices...);
  }

  template <typename... Indices>
  size_t get_1d_index(Indices... indices) const {
    return mdspan_.mapping()(indices...);
  }

  std::array<size_t, Rank> get_md_index(size_t linear_index) const {
    if (linear_index >= derived().size()) {
      throw std::out_of_range("Linear index out of range");
    }
    std::array<size_t, Rank> indices{};
    for (size_t i = 0; i < Rank; ++i) {
      indices[i] = (linear_index / stride_[i]) % shape_[i];
    }
    return indices;
  }

  size_t get_dim_index(size_t linear_index, size_t dim) const { return get_md_index(linear_index)[dim]; }

  void print() const
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      print_mdspan(mdspan_);
    }
  }

  const MdspanType& mdspan() const { return mdspan_; }

 protected:
  template <typename... Indices>
  void check_indices(Indices... indices) const {
    const size_t idx_array[Rank] = {static_cast<size_t>(indices)...};
    for (size_t i = 0; i < Rank; ++i) {
      if (idx_array[i] >= shape_[i]) {
        throw std::out_of_range(std::format("Index {} out of range (dim {}, size {})", idx_array[i], i, shape_[i]));
      }
    }
  }
};

}  // namespace md

#endif