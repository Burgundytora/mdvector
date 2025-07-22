#ifndef __MDVECTOR_MDENGINE_DYNAMIC_H__
#define __MDVECTOR_MDENGINE_DYNAMIC_H__

#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "expression_template/operator.h"
#include "mdspan.h"
#include "simd/allocator.h"
#include "simd/simd_function.h"
#include "subspan.h"
#include "subview.h"

namespace md {

template <class T, size_t Rank, class Layout = layout_right>
class EngineDynamic {
 protected:
  std::vector<T, auto_allocator<T>> data_;
  mdspan<T, Rank> view_;

 public:
  EngineDynamic() = default;

  explicit EngineDynamic(const std::array<std::size_t, Rank>& dims)
      : data_(calculate_size(dims)), view_(mdspan<T, Rank, Layout>(data_.data(), dims)) {
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "T must be trivial and standard-layout!");
  }

  ~EngineDynamic() = default;

  EngineDynamic(const EngineDynamic& other) : data_(other.data_), view_(data_.data(), other.view_.extents()) {}

  EngineDynamic(EngineDynamic&& other) noexcept
      : data_(std::move(other.data_)), view_(data_.data(), other.view_.extents()) {}

  EngineDynamic& operator=(const EngineDynamic& other) {
    if (this != &other) {
      data_ = other.data_;
      view_ = mdspan<T, Rank>(data_.data(), other.view_.extents());
    }
    return *this;
  }

  EngineDynamic& operator=(EngineDynamic&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      view_ = mdspan<T, Rank>(data_.data(), other.view_.extents());
    }
    return *this;
  }

  template <class... Indices>
  T& operator()(Indices... indices) {
    return view_(indices...);
  }

  template <class... Indices>
  const T& operator()(Indices... indices) const {
    return view_(indices...);
  }

  template <class... Indices>
  T& at(Indices... indices) {
    return view_.at(indices...);
  }

  template <class... Indices>
  const T& at(Indices... indices) const {
    return view_.at(indices...);
  }

  template <class... Indices>
  size_t& get_1d_index(Indices... indices) {
    return view_.get_1d_index(indices...);
  }

  static size_t calculate_size(const std::array<std::size_t, Rank>& dims) {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<>());
  }

  T* data() { return data_.data(); }

  const T* data() const { return data_.data(); }

  size_t used_size() const { return data_.size(); }

  size_t size() const { return data_.size(); }

  std::array<size_t, Rank> shapes() const { return view_.extents(); }

  std::array<size_t, Rank> extents() const { return view_.extents(); }

  void set_value(T val) { std::fill(data_.begin(), data_.end(), val); }

  void reset_shape(const std::array<std::size_t, Rank>& dims) {
    data_.resize(calculate_size(dims));
    view_ = mdspan<T, Rank>(data_.data(), dims);
  }

  // slice subspan with simd_ET
  template <class... Slices>
  subspan<T, Rank> view(Slices... slices) {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");
    auto slice_array = md::prepare_slices<Rank>(slices...);
    return subspan<T, Rank>(data_.data(), view_.extents(), slice_array);
  }

  // slice subview without simd_ET
  template <class... Slices>
  subspan<T, Rank> create_subview(Slices... slices) {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");
    auto slice_array = md::prepare_slices<Rank>(slices...);
    return subview<T, Rank>(data_.data(), view_.extents(), slice_array);
  }

  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() noexcept { return data_.data(); }
  iterator end() noexcept { return data_.data() + data_.size(); }
  const_iterator begin() const noexcept { return data_.data(); }
  const_iterator end() const noexcept { return data_.data() + data_.size(); }
  const_iterator cbegin() const noexcept { return data_.data(); }
  const_iterator cend() const noexcept { return data_.data() + data_.size(); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }
};

}  // namespace md

#endif  // __MDVECTOR_MDENGINE_DYNAMIC_H__