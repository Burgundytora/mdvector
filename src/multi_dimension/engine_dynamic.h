#ifndef __MDVECTOR_ENGINE_DYNAMIC_H__
#define __MDVECTOR_ENGINE_DYNAMIC_H__

#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "expression_template/operator.h"
#include "mdspan.h"
#include "simd/allocator.h"
#include "simd/simd_function.h"
#include "span.h"

namespace md {

template <class T, size_t Rank, class Layout = layout_right>
class engine_dynamic {
 protected:
  std::vector<T, auto_allocator<T>> data_;
  mdspan<T, Rank, Layout> mdspan_;

 public:
  engine_dynamic() = default;

  explicit engine_dynamic(const std::array<std::size_t, Rank>& dims)
      : data_(calculate_size(dims)), mdspan_(mdspan<T, Rank, Layout>(data_.data(), dims)) {
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "T must be trivial and standard-layout!");
  }

  ~engine_dynamic() = default;

  engine_dynamic(const engine_dynamic& other) : data_(other.data_), mdspan_(data_.data(), other.mdspan_.extents()) {}

  engine_dynamic(engine_dynamic&& other) noexcept
      : data_(std::move(other.data_)), mdspan_(data_.data(), other.mdspan_.extents()) {}

  engine_dynamic& operator=(const engine_dynamic& other) {
    if (this != &other) {
      data_ = other.data_;
      mdspan_ = mdspan<T, Rank>(data_.data(), other.mdspan_.extents());
    }
    return *this;
  }

  engine_dynamic& operator=(engine_dynamic&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      mdspan_ = mdspan<T, Rank>(data_.data(), other.mdspan_.extents());
    }
    return *this;
  }

  template <class... Indices>
  T& operator()(Indices... indices) {
    return mdspan_(indices...);
  }

  template <class... Indices>
  const T& operator()(Indices... indices) const {
    return mdspan_(indices...);
  }

#if defined(__cpp_multidimensional_subscript) || __cplusplus >= 202302L
  template <class... Indices>
  T& operator[](Indices... indices) {
    return mdspan_(indices...);
  }

  template <class... Indices>
  const T& operator[](Indices... indices) const {
    return mdspan_(indices...);
  }
#endif

  template <class... Indices>
  T& at(Indices... indices) {
    return mdspan_.at(indices...);
  }

  template <class... Indices>
  const T& at(Indices... indices) const {
    return mdspan_.at(indices...);
  }

  template <class... Indices>
  size_t& get_1d_index(Indices... indices) {
    return mdspan_.get_1d_index(indices...);
  }

  static size_t calculate_size(const std::array<std::size_t, Rank>& dims) {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<>());
  }

  T* data() { return data_.data(); }

  const T* data() const { return data_.data(); }

  size_t used_size() const { return data_.size(); }

  size_t size() const { return data_.size(); }

  std::array<size_t, Rank> shapes() const { return mdspan_.extents(); }

  std::array<size_t, Rank> extents() const { return mdspan_.extents(); }

  size_t extent(int index) const { return mdspan_.extents().at(index); }

  void set_value(T val) { std::fill(data_.begin(), data_.end(), val); }

  void reset_shape(const std::array<std::size_t, Rank>& dims) {
    data_.resize(calculate_size(dims));
    mdspan_ = mdspan<T, Rank>(data_.data(), dims);
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

#endif  // __MDVECTOR_ENGINE_DYNAMIC_H__