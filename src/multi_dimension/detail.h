#ifndef __MDVECTOR_SPAN_DETAIL_H__
#define __MDVECTOR_SPAN_DETAIL_H__

#include <array>
#include <cstddef>
#include <type_traits>

namespace md {

struct layout_right {};
struct layout_left {};

template <std::size_t Rank, class Layout = layout_right>
auto compute_strides(const std::array<std::size_t, Rank>& extents) {
  std::array<std::size_t, Rank> strides;
  if constexpr (std::is_same_v<Layout, layout_right>) {
    strides.back() = 1;
    for (int i = Rank - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * extents[i + 1];
    }
  } else {
    strides.front() = 1;
    for (int i = 1; i <= Rank - 1; i++) {
      strides[i] = strides[i - 1] * extents[i - 1];
    }
  }
  return strides;
}

template <std::size_t Rank>
constexpr std::size_t linear_index(const std::array<std::size_t, Rank>& strides,
                                   const std::array<std::size_t, Rank>& indices) {
  std::size_t idx = 0;
  for (std::size_t i = 0; i < Rank; ++i) {
    idx += indices[i] * strides[i];
  }
  return idx;
}

// subspan 闭区间
struct slice {
  std::ptrdiff_t start;
  std::ptrdiff_t end;
  bool is_all;

  slice(std::ptrdiff_t s = 0, std::ptrdiff_t e = 0, bool all = false) : start(s), end(e), is_all(all) {}
};

// 将负数索引转换为正数
static std::ptrdiff_t normalize_index(std::ptrdiff_t idx, std::ptrdiff_t dim_size) {
  return idx >= 0 ? idx : dim_size + idx;
}

// 创建全选切片
inline md::slice all() { return md::slice(0, 0, true); }

template <size_t Rank>
void check_slice_bounds(const std::array<md::slice, Rank>& slices, const std::array<std::size_t, Rank>& extents) {
  for (size_t i = 0; i < Rank; ++i) {
    if (slices[i].is_all) {
      continue;
    }

    // 处理负数索引（-1 表示最后一个元素）
    std::ptrdiff_t start = md::normalize_index(slices[i].start, extents[i]);
    std::ptrdiff_t end = md::normalize_index(slices[i].end, extents[i]);

    // 检查边界
    if (start < 0 || start >= static_cast<std::ptrdiff_t>(extents[i])) {
      throw std::out_of_range("subspan slice start out of range");
    }
    if (end < 0 || end >= static_cast<std::ptrdiff_t>(extents[i])) {
      throw std::out_of_range("subspan slice end out of range");
    }
    if (start > end) {  // 允许 start == end（单元素）
      throw std::invalid_argument("subspan slice start must <= end");
    }
  }
}

template <class SliceType>
md::slice convert_slice(SliceType&& slice_one) {
  if constexpr (std::is_same_v<std::decay_t<SliceType>, md::slice>) {
    return std::forward<SliceType>(slice_one);
  } else if constexpr (std::is_integral_v<std::decay_t<SliceType>>) {
    // 整数索引转换为单元素切片
    return md::slice(static_cast<std::ptrdiff_t>(slice_one), static_cast<std::ptrdiff_t>(slice_one), false);
  } else {
    static_assert(sizeof(SliceType) == 0, "Unsupported slice type");
  }
}

template <size_t Rank, class... Slices>
std::array<md::slice, Rank> prepare_slices(Slices... slices) {
  std::array<md::slice, Rank> result;
  size_t i = 0;

  // 使用折叠表达式处理每个切片
  ((result[i++] = md::convert_slice(slices)), ...);

  return result;
}

}  // namespace md

#endif  // MDVECTOR_SPAN_DETAIL_H_