#ifndef __MDVECTOR_MULTI_DIM__
#define __MDVECTOR_MULTI_DIM__

#include "../concepts/base_concept.h"
#include "mdspan_impl.h"
#include "mdspan_print.h"

#include <format>

namespace md {

// ============================================================================
// 1. multi_dim_dynamic - 用于 vector 和 span（动态维度，连续内存）
// ============================================================================
template <typename Derived, typename T, size_t Rank, typename Layout>
class multi_dim_dynamic {
 protected:
  std::array<size_t, Rank> shape_;
  std::mdspan<T, std::dextents<size_t, Rank>, Layout> mdspan_;

  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

 public:
  multi_dim_dynamic() = default;

  template <size_t... Indices>
  multi_dim_dynamic(const std::array<size_t, Rank>& shape, std::index_sequence<Indices...>) : shape_(shape) {
    mdspan_ = std::mdspan<T, std::dextents<size_t, Rank>, Layout>(derived().data(), shape_[Indices]...);
  }

  template <size_t... Indices>
  void init_mdspan(std::index_sequence<Indices...>) {
    mdspan_ = std::mdspan<T, std::dextents<size_t, Rank>, Layout>(derived().data(), shape_[Indices]...);
  }

  auto extents() const noexcept { return shape_; }
  size_t extent(size_t dim) const noexcept { return shape_[dim]; }

  auto constexpr rank() const noexcept { return Rank; }

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
    size_t remaining = linear_index;
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
      }
    } else {
      for (size_t i = 0; i < Rank; ++i) {
        indices[i] = remaining % shape_[i];
        remaining /= shape_[i];
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

  auto& mdspan() { return mdspan_; }
  const auto& mdspan() const { return mdspan_; }

 protected:
  template <typename... Indices>
  void check_indices(Indices... indices) const {
    if (mdspan_.empty()) {
      throw std::logic_error("multi dim dynamic not initialized");
    }

    const size_t idx_array[Rank] = {static_cast<size_t>(indices)...};
    for (size_t i = 0; i < Rank; ++i) {
      if (idx_array[i] >= shape_[i]) {
        throw std::out_of_range(std::format("Index {} out of range (dim {}, size {})", idx_array[i], i, shape_[i]));
      }
    }
  }
};

// ============================================================================
// 2. multi_dim_static - 用于 array（编译期固定维度）
// ============================================================================
template <typename Derived, typename T, typename Layout, size_t... Lengths>
class multi_dim_static {
 public:
  static constexpr size_t Rank = sizeof...(Lengths);
  static constexpr std::array<size_t, Rank> shape_ = {Lengths...};

 protected:
  std::mdspan<T, std::extents<size_t, Lengths...>, Layout> mdspan_{derived().data()};

  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

 public:
  multi_dim_static() = default;

  static constexpr auto extents() noexcept { return shape_; }
  static constexpr size_t extent(size_t dim) noexcept { return shape_[dim]; }

  auto constexpr rank() const noexcept { return Rank; }

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
  constexpr size_t get_1d_index(Indices... indices) const {
    return mdspan_.mapping()(indices...);
  }

  constexpr std::array<size_t, Rank> get_md_index(size_t linear_index) const {
    std::array<size_t, Rank> indices{};
    size_t remaining = linear_index;
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((indices[Rank - 1 - Is] = remaining % Lengths, remaining /= Lengths), ...);
      }(std::index_sequence<Lengths...>{});
    } else {
      [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((indices[Is] = remaining % Lengths, remaining /= Lengths), ...);
      }(std::index_sequence<Lengths...>{});
    }
    return indices;
  }

  constexpr size_t get_dim_index(size_t linear_index, size_t dim) const { return get_md_index(linear_index)[dim]; }

  void print() const
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      print_mdspan(mdspan_);
    }
  }

  auto& mdspan() { return mdspan_; }
  const auto& mdspan() const { return mdspan_; }

 protected:
  template <typename... Indices>
  constexpr void check_indices(Indices... indices) const {
    const size_t idx_array[Rank] = {static_cast<size_t>(indices)...};
    for (size_t i = 0; i < Rank; ++i) {
      if (idx_array[i] >= shape_[i]) {
        throw std::out_of_range(std::format("Index {} out of range (dim {}, size {})", idx_array[i], i, shape_[i]));
      }
    }
  }
};

// ============================================================================
// 3. multi_dim_stride - 用于 view（跨步，不连续）
// ============================================================================
template <typename Derived, typename T, size_t Rank>
class multi_dim_stride {
 protected:
  std::array<size_t, Rank> shape_;
  std::array<size_t, Rank> stride_;
  std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride> mdspan_;

  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

 public:
  multi_dim_stride() = default;

  multi_dim_stride(const std::array<size_t, Rank>& shape, const std::array<size_t, Rank>& stride) {
    shape_ = shape;
    stride_ = stride;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      mdspan_ = std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride>(
          derived().data(),
          std::layout_stride::mapping(std::dextents<size_t, Rank>(shape[Is]...),  // ✅ 展开为 shape[0], shape[1], ...
                                      stride));
    }(std::make_index_sequence<Rank>{});
  }

  void init_mdspan(const std::array<size_t, Rank>& shape, const std::array<size_t, Rank>& stride) {
    shape_ = shape;
    stride_ = stride;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      mdspan_ = std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride>(
          derived().data(),
          std::layout_stride::mapping(std::dextents<size_t, Rank>(shape[Is]...),  // ✅ 展开为 shape[0], shape[1], ...
                                      stride));
    }(std::make_index_sequence<Rank>{});
  }

  auto extents() const noexcept { return shape_; }
  size_t extent(size_t dim) const noexcept { return shape_[dim]; }

  auto strides() const noexcept { return stride_; }
  size_t stride(size_t dim) const noexcept { return stride_[dim]; }

  auto constexpr rank() const noexcept { return Rank; }

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

  const auto& mdspan() const { return mdspan_; }

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

#endif  // __MDVECTOR_MULTI_DIM__