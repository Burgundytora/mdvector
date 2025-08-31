#ifndef __MDVECTOR_DETAIL_H__
#define __MDVECTOR_DETAIL_H__

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

// 闭区间
struct slice {
  std::ptrdiff_t start;
  std::ptrdiff_t end;
  bool is_all;

  slice(std::ptrdiff_t s = 0, std::ptrdiff_t e = 0, bool all = false) : start(s), end(e), is_all(all) {}
};

// 将python风格负数索引 转换为正数
static std::ptrdiff_t normalize_index(std::ptrdiff_t idx, std::ptrdiff_t dim_size) {
  return idx >= 0 ? idx : dim_size + idx;
}

// 全选切片
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
      throw std::out_of_range("span slice start out of range");
    }
    if (end < 0 || end >= static_cast<std::ptrdiff_t>(extents[i])) {
      throw std::out_of_range("span slice end out of range");
    }
    if (start > end) {  // 允许 start == end（单元素）
      throw std::invalid_argument("span slice start must <= end");
    }
  }
}

template <size_t Rank, class Layout = layout_right>
bool check_slice_contiguous(std::array<std::size_t, Rank> ori, std::array<slice, Rank> slice,
                            std::array<bool, Rank> is_single) {
  bool contiguous = true;
  // 3:所有 2:部分 1:单个
  // 判断连续性
  // 上一层是3 这层可以是123
  // 上一层是2 这一层必须是1
  // 上一层是1 这一层必须是1
  int this_level = 3;
  int last_level = 3;
  int loop_i, step, end_val;
  if constexpr (std::is_same_v<Layout, layout_right>) {
    loop_i = ori.size() - 1;
    step = -1;
    end_val = -1;
  } else {
    loop_i = 0;
    step = 1;
    end_val = ori.size();
  }

  for (; loop_i != end_val; loop_i += step) {
    int slice_length = slice.at(loop_i).end - slice.at(loop_i).start + 1;
    if (slice.at(loop_i).is_all || slice_length == ori.at(loop_i)) {
      this_level = 3;
    } else if (slice_length == 1) {
      this_level = 1;
    } else {
      this_level = 2;
    }

    if (last_level == 3) {
      contiguous = true;
    } else {
      if (this_level != 1) {
        contiguous = false;
      } else {
        contiguous = true;
      }
    }
    last_level = this_level;

    if (!contiguous) {
      contiguous = false;
      break;
    }
  }
  return contiguous;
}

//////
// 辅助类型：判断是否是整数类型
template <class T>
struct is_integral_slice : std::false_type {};

template <class T>
struct is_integral_slice<std::integral_constant<T, T{}>> : std::true_type {};

template <class T>
constexpr bool is_integral_slice_v = is_integral_slice<T>::value;

// 计算新维度（Rank）的元函数
template <class... Slices>
struct compressed_rank;

template <>
struct compressed_rank<> : std::integral_constant<std::size_t, 0> {};

template <class First, class... Rest>
struct compressed_rank<First, Rest...>
    : std::integral_constant<std::size_t, (!std::is_integral_v<std::decay_t<First>>)+compressed_rank<Rest...>::value> {
};

template <class... Slices>
constexpr std::size_t compressed_rank_v = compressed_rank<Slices...>::value;

// 转换切片并收集信息
template <std::size_t Rank, class... Slices>
auto prepare_slices(std::array<std::size_t, Rank> extents, Slices... slices) {
  std::array<slice, Rank> result;
  std::array<bool, Rank> is_integral{};  // 标记哪些维度是整数索引

  std::size_t i = 0;
  // 处理每个切片
  ((result[i] = convert_slice(extents[i], slices), is_integral[i] = std::is_integral_v<std::decay_t<decltype(slices)>>,
    i++),
   ...);

  return std::make_pair(result, is_integral);
}

template <class SliceType>
md::slice convert_slice(int this_dim_size, SliceType&& slice_one) {
  if constexpr (std::is_same_v<std::decay_t<SliceType>, md::slice>) {
    return std::forward<SliceType>(slice_one);
  } else if constexpr (std::is_integral_v<std::decay_t<SliceType>>) {
    // 整数索引转换为单元素切片
    std::ptrdiff_t normolize_index = md::normalize_index(slice_one, this_dim_size);
    return md::slice(static_cast<std::ptrdiff_t>(normolize_index), static_cast<std::ptrdiff_t>(normolize_index), false);
  } else {
    static_assert(sizeof(SliceType) == 0, "Unsupported slice type");
  }
}

}  // namespace md

#endif  // MDVECTOR_DETAIL_H_