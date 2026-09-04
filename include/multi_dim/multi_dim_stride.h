#pragma once

#include "mdspan_print.h"

#include <array>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <tuple>

namespace md {

// std::layout_stride only accepts mappings that satisfy its uniqueness and
// stride-ordering preconditions. A general slice can be perfectly valid while
// not satisfying those preconditions (and a negative stride never does), so a
// view must not use std::mdspan as its internal address calculator.
template <typename T, size_t Rank>
class strided_mdspan_adapter {
 public:
  using element_type = T;
  using index_type = std::ptrdiff_t;
  using size_type = size_t;
  using reference = T&;

  class mapping_type {
   public:
    mapping_type(const std::array<size_t, Rank>& shape,
                 const std::array<std::ptrdiff_t, Rank>& stride) noexcept
        : shape_(&shape), stride_(&stride) {}

    template <typename... Indices>
    std::ptrdiff_t operator()(Indices... indices) const noexcept {
      static_assert(sizeof...(Indices) == Rank);
      const std::array<std::ptrdiff_t, Rank> index{
          static_cast<std::ptrdiff_t>(indices)...};
      std::ptrdiff_t offset = 0;
      for (size_t d = 0; d < Rank; ++d) offset += index[d] * (*stride_)[d];
      return offset;
    }

    size_t extent(size_t dim) const noexcept { return (*shape_)[dim]; }
    std::ptrdiff_t stride(size_t dim) const noexcept { return (*stride_)[dim]; }

   private:
    const std::array<size_t, Rank>* shape_;
    const std::array<std::ptrdiff_t, Rank>* stride_;
  };

  strided_mdspan_adapter() = default;

  strided_mdspan_adapter(T* data, const std::array<size_t, Rank>& shape,
                         const std::array<std::ptrdiff_t, Rank>& stride) noexcept
      : data_(data), shape_(shape), stride_(stride) {}

  void reset(T* data, const std::array<size_t, Rank>& shape,
             const std::array<std::ptrdiff_t, Rank>& stride) noexcept {
    data_ = data;
    shape_ = shape;
    stride_ = stride;
  }

  static constexpr size_t rank() noexcept { return Rank; }
  size_t extent(size_t dim) const noexcept { return shape_[dim]; }
  std::ptrdiff_t stride(size_t dim) const noexcept { return stride_[dim]; }
  const auto& extents() const noexcept { return shape_; }
  const auto& strides() const noexcept { return stride_; }

  size_t size() const noexcept {
    size_t result = 1;
    for (size_t extent : shape_) result *= extent;
    return result;
  }

  bool empty() const noexcept { return size() == 0; }
  T* data_handle() const noexcept { return data_; }
  mapping_type mapping() const noexcept { return mapping_type(shape_, stride_); }

  template <typename... Indices>
  T& operator()(Indices... indices) const noexcept {
    return data_[mapping()(indices...)];
  }

  template <typename... Indices>
  T& operator[](Indices... indices) const noexcept {
    return operator()(indices...);
  }

  T& logical_at(size_t linear) const noexcept {
    std::array<size_t, Rank> indices{};
    for (size_t d = Rank; d-- > 0;) {
      indices[d] = linear % shape_[d];
      linear /= shape_[d];
    }
    return std::apply([this](auto... index) -> T& { return operator()(index...); }, indices);
  }

 private:
  T* data_ = nullptr;
  std::array<size_t, Rank> shape_{};
  std::array<std::ptrdiff_t, Rank> stride_{};
};

// Multidimensional indexing for a possibly non-contiguous view.
template <typename Derived, typename T, size_t Rank>
class multi_dim_stride {
 protected:
  std::array<size_t, Rank> shape_{};
  std::array<std::ptrdiff_t, Rank> stride_{};
  strided_mdspan_adapter<T, Rank> mdspan_{};

  Derived& derived() noexcept { return static_cast<Derived&>(*this); }
  const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

 public:
  multi_dim_stride() = default;

  multi_dim_stride(const std::array<size_t, Rank>& shape,
                   const std::array<std::ptrdiff_t, Rank>& stride)
      : shape_(shape), stride_(stride) {}

  void init_mdspan(const std::array<size_t, Rank>& shape,
                   const std::array<std::ptrdiff_t, Rank>& stride) noexcept {
    shape_ = shape;
    stride_ = stride;
    mdspan_.reset(derived().data(), shape_, stride_);
  }

  auto extents() const noexcept { return shape_; }
  size_t extent(size_t dim) const noexcept { return shape_[dim]; }

  auto strides() const noexcept { return stride_; }
  std::ptrdiff_t stride(size_t dim) const noexcept { return stride_[dim]; }

  static constexpr size_t rank() noexcept { return Rank; }

  template <typename... Indices>
  T& operator()(Indices... indices) noexcept {
    static_assert(sizeof...(Indices) == Rank);
    return derived().data()[get_1d_index(indices...)];
  }

  template <typename... Indices>
  const T& operator()(Indices... indices) const noexcept {
    static_assert(sizeof...(Indices) == Rank);
    return derived().data()[get_1d_index(indices...)];
  }

  template <typename... Indices>
  T& operator[](Indices... indices) noexcept {
    return operator()(indices...);
  }

  template <typename... Indices>
  const T& operator[](Indices... indices) const noexcept {
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
  std::ptrdiff_t get_1d_index(Indices... indices) const noexcept {
    static_assert(sizeof...(Indices) == Rank);
    const std::array<std::ptrdiff_t, Rank> index{
        static_cast<std::ptrdiff_t>(indices)...};
    std::ptrdiff_t offset = 0;
    for (size_t d = 0; d < Rank; ++d) offset += index[d] * stride_[d];
    return offset;
  }

  std::array<size_t, Rank> get_md_index(size_t linear_index) const {
    if (linear_index >= derived().size()) {
      throw std::out_of_range("Linear index out of range");
    }
    std::array<size_t, Rank> indices{};
    for (size_t i = Rank; i-- > 0;) {
      indices[i] = linear_index % shape_[i];
      linear_index /= shape_[i];
    }
    return indices;
  }

  size_t get_dim_index(size_t linear_index, size_t dim) const {
    if (dim >= Rank) throw std::out_of_range("Dimension out of range");
    return get_md_index(linear_index)[dim];
  }

  void print() const
    requires Printable<T>
  {
    if (!mdspan_.empty()) print_mdspan(mdspan_);
  }

  const auto& mdspan() const noexcept { return mdspan_; }

 protected:
  template <typename... Indices>
  void check_indices(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank);
    const std::array<std::ptrdiff_t, Rank> index{
        static_cast<std::ptrdiff_t>(indices)...};
    for (size_t i = 0; i < Rank; ++i) {
      if (index[i] < 0 || static_cast<size_t>(index[i]) >= shape_[i]) {
        throw std::out_of_range(
            std::format("Index {} out of range (dim {}, size {})", index[i], i, shape_[i]));
      }
    }
  }
};

}  // namespace md
