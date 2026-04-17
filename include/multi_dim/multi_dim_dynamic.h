#ifndef __MDVECTOR_MULTI_DIM_DYNAMIC__
#define __MDVECTOR_MULTI_DIM_DYNAMIC__

#include "mdspan_print.h"

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

}  // namespace md

#endif  // __MDVECTOR_MULTI_DIM_DYNAMIC__