#ifndef __MDSPAN_H__

#include <array>
#include <cstddef>
#include <type_traits>

// 布局标签
struct layout_right {};  // 行主序
struct layout_left {};   // 列主序

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

// 处理Python风格的切片参数
struct Slice {
  std::ptrdiff_t start;
  std::ptrdiff_t stop;
  std::ptrdiff_t step;

  Slice(std::ptrdiff_t s = 0, std::ptrdiff_t e = 0, std::ptrdiff_t st = 1) : start(s), stop(e), step(st) {}
};

// 转换负索引为正索引
constexpr std::size_t normalize_index(std::ptrdiff_t idx, std::size_t size) {
  if (idx < 0) {
    idx += size;
    if (idx < 0) throw std::out_of_range("Index out of range");
  }
  if (static_cast<std::size_t>(idx) >= size) {
    throw std::out_of_range("Index out of range");
  }
  return static_cast<std::size_t>(idx);
}

// 计算切片后的实际大小
constexpr std::size_t compute_slice_size(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step,
                                         std::size_t dim_size) {
  if (step == 0) throw std::invalid_argument("Step cannot be zero");

  if (step > 0) {
    start = (start < 0) ? std::max<std::ptrdiff_t>(0, start + dim_size) : std::min<std::ptrdiff_t>(start, dim_size);
    stop = (stop < 0) ? std::max<std::ptrdiff_t>(0, stop + dim_size) : std::min<std::ptrdiff_t>(stop, dim_size);
    return (start < stop) ? (stop - start + step - 1) / step : 0;
  } else {
    start = (start < 0) ? start + dim_size : std::min<std::ptrdiff_t>(start, dim_size - 1);
    stop = (stop < 0) ? stop + dim_size : std::min<std::ptrdiff_t>(stop, -1);
    return (start > stop) ? (start - stop - step - 1) / (-step) : 0;
  }
}

}  // namespace detail

// 动态维度的mdspan
template <typename T, size_t Dims, typename Layout = layout_right>
class mdspan {
 public:
  using element_type = T;

  // 构造函数
  constexpr mdspan() noexcept = default;

  // 从指针和维度array构造
  constexpr mdspan(T* data, const std::array<std::size_t, Dims>& extents)
      : data_(data), extents_(extents), strides_(detail::compute_strides(extents)) {}

  // 从容器构造
  template <typename Container>
  constexpr mdspan(Container& c, const std::array<std::size_t, Dims>& extents) : mdspan(c.data(), extents) {}

  // 元素访问
  template <typename... Indices>
  constexpr T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "Number of indices must match Dims");
    std::array<std::size_t, Dims> idxs{static_cast<std::size_t>(indices)...};
    return data_[detail::linear_index(strides_, idxs)];
  }

  // Python风格的subspan实现
  template <typename... SliceSpecs>
  auto subspan(SliceSpecs... slices) const {
    static_assert(sizeof...(SliceSpecs) == Dims, "Number of slices must match rank");

    std::array<std::size_t, Dims> offsets;
    std::array<std::size_t, Dims> counts;
    std::array<std::ptrdiff_t, Dims> steps;

    process_slices(slices..., offsets, counts, steps, std::make_index_sequence<Dims>{});

    // 创建带步长的视图（简化版，实际需要更复杂的实现）
    return create_strided_view(offsets, counts, steps);
  }

  // 属性访问
  constexpr std::size_t rank() const noexcept { return Dims; }
  std::size_t extent(std::size_t r) const { return extents_[r]; }
  T* data() const noexcept { return data_; }
  std::array<std::size_t, Dims> shape() const { return extents_; }
  std::array<std::size_t, Dims> extents() const { return extents_; }

 private:
  // 处理各种切片参数
  template <std::size_t... Is, typename... SliceSpecs>
  void process_slices(SliceSpecs... slices, std::array<std::size_t, Dims>& offsets,
                      std::array<std::size_t, Dims>& counts, std::array<std::ptrdiff_t, Dims>& steps,
                      std::index_sequence<Is...>) const {
    (
        [&] {
          using SliceType = std::tuple_element_t<Is, std::tuple<SliceSpecs...>>;

          if constexpr (std::is_integral_v<SliceType>) {
            // 整数索引 - 降维
            offsets[Is] = detail::normalize_index(slices, extents_[Is]);
            counts[Is] = 1;
            steps[Is] = 1;
          } else if constexpr (std::is_same_v<SliceType, detail::Slice>) {
            // 显式切片对象
            const auto& slice = slices;
            counts[Is] = detail::compute_slice_size(slice.start, slice.stop, slice.step, extents_[Is]);

            if (slice.step > 0) {
              offsets[Is] = (slice.start < 0) ? std::max<std::ptrdiff_t>(0, slice.start + extents_[Is])
                                              : std::min<std::ptrdiff_t>(slice.start, extents_[Is]);
            } else {
              offsets[Is] = (slice.start < 0) ? slice.start + extents_[Is]
                                              : std::min<std::ptrdiff_t>(slice.start, extents_[Is] - 1);
            }

            steps[Is] = slice.step;
          } else {
            // 其他情况视为全切片 [:]
            offsets[Is] = 0;
            counts[Is] = extents_[Is];
            steps[Is] = 1;
          }
        }(),
        ...);
  }

  // 创建带步长的视图（简化实现）
  auto create_strided_view(const std::array<std::size_t, Dims>& offsets, const std::array<std::size_t, Dims>& counts,
                           const std::array<std::ptrdiff_t, Dims>& steps) const {
    // 计算新指针位置
    std::size_t offset = 0;
    for (std::size_t i = 0; i < Dims; ++i) {
      offset += offsets[i] * strides_[i];
    }

    // 计算新的strides（考虑步长）
    std::array<std::size_t, Dims> new_strides;
    for (std::size_t i = 0; i < Dims; ++i) {
      new_strides[i] = strides_[i] * steps[i];
    }

    // 返回新的mdspan（简化版，实际可能需要更复杂的视图类型）
    return mdspan(data_ + offset, counts);
  }

 private:
  T* data_ = nullptr;
  std::array<std::size_t, Dims> extents_;
  std::array<std::size_t, Dims> strides_;
  static constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);
};

#endif  // __MDSPAN_H__