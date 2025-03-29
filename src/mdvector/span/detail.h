#ifndef MDVECTOR_SPAN_DETAIL_H_
#define MDVECTOR_SPAN_DETAIL_H_

#include <array>
#include <cstddef>
#include <type_traits>

// 计算维度相关信息
namespace detail {

// 计算strides (行主序)
template <std::size_t Rank>
constexpr auto compute_strides(const std::array<std::size_t, Rank>& extents) {
  std::array<std::size_t, Rank> strides;
  strides.back() = 1;
  for (int i = Rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * extents[i + 1];
  }
  return strides;
}

// 计算线性索引
template <std::size_t Rank>
constexpr std::size_t linear_index(const std::array<std::size_t, Rank>& strides,
                                   const std::array<std::size_t, Rank>& indices) {
  std::size_t idx = 0;
  for (std::size_t i = 0; i < Rank; ++i) {
    idx += indices[i] * strides[i];
  }
  return idx;
}

// subspan 切片 闭区间
struct Slice {
  std::ptrdiff_t start;
  std::ptrdiff_t end;
  bool is_all;

  Slice(std::ptrdiff_t s = 0, std::ptrdiff_t e = 0, bool all = false) : start(s), end(e), is_all(all) {}
};

inline Slice all() {
  return Slice(0, 0, true);  // 创建全选切片
}

}  // namespace detail

#endif  // MDVECTOR_SPAN_DETAIL_H_