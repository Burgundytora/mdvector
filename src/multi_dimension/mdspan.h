#ifndef __MDVECTOR_MDSPAN_H__
#define __MDVECTOR_MDSPAN_H__

#include "detail.h"

template <class T, size_t Rank, class Layout = md::layout_right>
class mdspan {
 public:
  constexpr mdspan() noexcept = default;

  constexpr mdspan(T* data, const std::array<std::size_t, Rank>& extents)
      : data_(data), extents_(extents), strides_(md::compute_strides<Rank, Layout>(extents)) {
    size_ = 1;
    for (auto s : extents) {
      size_ *= s;
    }
  }

  template <class... Indices>
  void check_bounds(Indices... indices) const {
    constexpr std::size_t rank = sizeof...(Indices);
    std::array<std::size_t, rank> idxs{static_cast<std::size_t>(indices)...};
    for (std::size_t i = 0; i < rank; ++i) {
      if (idxs[i] >= extent(i)) {
        throw std::out_of_range("multi dimension subscript out of range");
      }
    }
  }

  template <class... Indices>
  constexpr T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
    std::array<std::size_t, Rank> idxs{static_cast<std::size_t>(indices)...};
    return data_[md::linear_index(strides_, idxs)];
  }

  template <class... Indices>
  constexpr T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
    check_bounds(indices...);
    std::array<std::size_t, Rank> idxs{static_cast<std::size_t>(indices)...};
    return data_[md::linear_index(strides_, idxs)];
  }

  template <class... Indices>
  constexpr T& get_1d_index(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
    check_bounds(indices...);
    std::array<std::size_t, Rank> idxs{static_cast<std::size_t>(indices)...};
    return md::linear_index(strides_, idxs);
  }

  constexpr std::size_t rank() const noexcept { return Rank; }

  std::size_t shape(std::size_t r) const { return extents_.at(r); }

  std::size_t extent(std::size_t r) const { return extents_.at(r); }

  T* data() { return data_; }

  const T* data() const { return data_; }

  std::array<std::size_t, Rank> shapes() const { return extents_; }

  std::array<std::size_t, Rank> extents() const { return extents_; }

 protected:
  T* data_ = nullptr;
  size_t size_ = 1;
  std::array<std::size_t, Rank> extents_;
  std::array<std::size_t, Rank> strides_;
};

#endif  // __MDVECTOR_MDSPAN_H__