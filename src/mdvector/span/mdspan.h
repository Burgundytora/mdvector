#ifndef __MDVECTOR_MDSPAN_H__
#define __MDVECTOR_MDSPAN_H__

#include "detail.h"

// 布局标签
struct layout_right {};  // 行主序
struct layout_left {};   // 列主序

// 动态维度的mdspan
template <typename T, size_t Rank, typename Layout = layout_right>
class mdspan {
 public:
  // 构造函数
  constexpr mdspan() noexcept = default;

  // 从指针和维度array构造
  constexpr mdspan(T* data, const std::array<std::size_t, Rank>& extents)
      : data_(data), extents_(extents), strides_(md::compute_strides(extents)) {
    size_ = 1;
    for (auto s : extents) {
      size_ *= s;
    }
  }

  // 从容器构造
  template <typename Container>
  constexpr mdspan(Container& c, const std::array<std::size_t, Rank>& extents) : mdspan(c.data(), extents) {
    size_ = 1;
    for (auto s : extents) {
      size_ *= s;
    }
  }

  // 越界检查
  template <typename... Indices>
  void check_bounds(Indices... indices) const {
    constexpr std::size_t rank = sizeof...(Indices);
    std::array<std::size_t, rank> idxs{static_cast<std::size_t>(indices)...};
    for (std::size_t i = 0; i < rank; ++i) {
      if (idxs[i] >= extent(i)) {
        throw std::out_of_range("multi dimension subscript out of range");
      }
    }
  }

  // 元素访问
  template <typename... Indices>
  constexpr T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
    std::array<std::size_t, Rank> idxs{static_cast<std::size_t>(indices)...};
    return data_[md::linear_index(strides_, idxs)];
  }

  // 安全访问
  template <typename... Indices>
  constexpr T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
    check_bounds(indices...);
    std::array<std::size_t, Rank> idxs{static_cast<std::size_t>(indices)...};
    return data_[md::linear_index(strides_, idxs)];
  }

  // 获取一维索引
  template <typename... Indices>
  constexpr T& get_1d_index(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match Rank");
    check_bounds(indices...);
    std::array<std::size_t, Rank> idxs{static_cast<std::size_t>(indices)...};
    return md::linear_index(strides_, idxs);
  }

  // 属性访问
  constexpr std::size_t rank() const noexcept { return Rank; }

  std::size_t shape(std::size_t r) const { return extents_[r]; }

  std::size_t extent(std::size_t r) const { return extents_[r]; }

  T* data() { return data_; }

  const T* data() const { return data_; }

  std::array<std::size_t, Rank> shape() const { return extents_; }

  std::array<std::size_t, Rank> extents() const { return extents_; }

 protected:
  T* data_ = nullptr;
  size_t size_ = 1;
  std::array<std::size_t, Rank> extents_;
  std::array<std::size_t, Rank> strides_;
};

#endif  // __MDVECTOR_MDSPAN_H__