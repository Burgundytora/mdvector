#pragma once

#include "mdspan_print.h"

namespace md {

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

}  // namespace md
