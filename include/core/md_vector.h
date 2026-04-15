#ifndef __MDVECTOR_MD_VECTOR__
#define __MDVECTOR_MD_VECTOR__

#include "md_span.h"
#include "md_view.h"

#include "../storage/storage.h"
#include "../iterator/iterator.h"
#include "../multi_dim/multi_dim.h"
#include "../multi_dim/slice.h"
#include "../expression/expression.h"

namespace md {

template <typename T, size_t Rank, typename Layout = std::layout_right>
class vector final : public base_expr<vector<T, Rank, Layout>, T>,
                     public heap_storage<T>,
                     public multi_dim_dynamic<vector<T, Rank, Layout>, T, Rank, Layout>,
                     public iterator_contiguous<vector<T, Rank, Layout>, T>,
                     public fill_op<vector<T, Rank, Layout>, T>,
                     public expression<vector<T, Rank, Layout>, T, aligned_policy> {
 public:
  using simd_policy = aligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

  using BaseExpr = base_expr<vector<T, Rank, Layout>, T>;
  using Storage = heap_storage<T>;
  using MultiDim = multi_dim_dynamic<vector<T, Rank, Layout>, T, Rank, Layout>;
  using Iterator = iterator_contiguous<vector<T, Rank, Layout>, T>;
  using FillOps = fill_op<vector<T, Rank, Layout>, T>;
  using Expr = expression<vector<T, Rank, Layout>, T, aligned_policy>;

  // ============ 构造函数 ============

  vector() = default;

  explicit vector(const std::array<size_t, Rank>& shape) : Storage(calculate_size(shape)) {
    MultiDim::shape_ = shape;
    MultiDim::init_mdspan(std::make_index_sequence<Rank>{});
  }

  template <typename... Sizes>
    requires(sizeof...(Sizes) == Rank && (std::convertible_to<Sizes, size_t> && ...))
  explicit vector(Sizes... sizes) : vector(std::array<size_t, Rank>{static_cast<size_t>(sizes)...}) {}

  // 从表达式构造
  template <typename E>
  vector(const base_expr<E, T>& expr)
    requires Numeric<T>
  {
    MultiDim::shape_ = expr.extents();
    Storage::resize(calculate_size(MultiDim::shape_));
    MultiDim::init_mdspan(std::make_index_sequence<Rank>{});
    expr.template eval_to<>(*this);
  }

  // 拷贝/移动
  vector(const vector& other) : Storage(other.size()), MultiDim() {
    MultiDim::shape_ = other.shape_;
    MultiDim::init_mdspan(std::make_index_sequence<Rank>{});
    std::copy(other.begin(), other.end(), this->begin());
  }

  vector(vector&& other) noexcept = default;

  vector& operator=(const vector& other) {
    if (this != &other) {
      Storage::resize(other.size());
      MultiDim::shape_ = other.extents();
      MultiDim::init_mdspan(std::make_index_sequence<Rank>{});
      std::copy(other.begin(), other.end(), this->begin());
    }
    return *this;
  }

  vector& operator=(vector&& other) noexcept = default;

  // ============ 形状修改 ============

  void set_shape(const std::array<size_t, Rank>& shape) {
    if (shape == MultiDim::shape_ && !MultiDim::mdspan_.empty()) {
      return;
    }
    Storage::resize(calculate_size(shape));
    MultiDim::shape_ = shape;
    MultiDim::init_mdspan(std::make_index_sequence<Rank>{});
  }

  template <typename... Sizes>
    requires(sizeof...(Sizes) == Rank && (std::convertible_to<Sizes, size_t> && ...))
  void set_shape(Sizes... sizes) {
    set_shape(std::array<size_t, Rank>{static_cast<size_t>(sizes)...});
  }

  // ============ 转发接口 ============

  using Storage::data;
  using Storage::size;
  using Storage::used_size;
  using Storage::capacity;

  using MultiDim::extents;
  using MultiDim::extent;
  using MultiDim::rank;
  using MultiDim::operator();
  using MultiDim::operator[];
  using MultiDim::at;
  using MultiDim::get_1d_index;
  using MultiDim::get_md_index;
  using MultiDim::get_dim_index;
  using MultiDim::print;
  using MultiDim::mdspan;
  using MultiDim::check_indices;

  using Iterator::begin;
  using Iterator::end;
  using Iterator::cbegin;
  using Iterator::cend;
  using Iterator::rbegin;
  using Iterator::rend;
  using Iterator::crbegin;
  using Iterator::crend;

  using FillOps::fill;
  using FillOps::set_zeros;
  using FillOps::set_ones;
  using FillOps::set_arange;
  using FillOps::set_random_uniform;
  using FillOps::set_random_normal;

  using Expr::operator=;
  using Expr::operator+=;
  using Expr::operator-=;
  using Expr::operator*=;
  using Expr::operator/=;
  using Expr::operator-;
  using Expr::operator+;

  // ============ SIMD IO ============

  template <typename T2>
  auto load_simd(size_t i) const noexcept
    requires Numeric<T>
  {
    return simd_policy::template load<T2>(data() + i);
  }

  template <typename T2>
    requires Numeric<T>
  void store_simd(size_t i, typename simd<T2>::const_ref_type val) noexcept {
    simd_policy::template store<T2>(data() + i, val);
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 创建span 内存连续视图
  template <typename... Slices>
  auto span(Slices... slices) {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");

    constexpr std::size_t NewRank = compressed_rank_v<Slices...>;

    auto [slice_array, is_integral] = prepare_slices<Rank>(extents(), slices...);

    // 检查越界
    check_slice_bounds<Rank>(slice_array, extents());

    // 检查内存连续
    if (!check_slice_contiguous<Rank, layout_type>(extents(), slice_array, is_integral)) {
      throw std::runtime_error("span slices must result in contiguous memory");
    }

    // 计算新的extents
    std::array<std::size_t, NewRank> new_extents;
    std::size_t new_idx = 0;

    for (int i = 0; i < Rank; ++i) {
      if (!is_integral[i]) {  // 只保留非整数索引的维度
        const auto& s = slice_array[i];
        std::ptrdiff_t start = normalize_index(s.start, extent(i));
        std::ptrdiff_t end = normalize_index(s.end, extent(i));
        new_extents[new_idx++] = s.is_all ? extent(i) : (end - start + 1);
        if (s.step != 1) {
          throw std::invalid_argument("span slice's step must be 1.");
        }
      }
    }

    // 计算新的数据指针偏移
    std::size_t offset = calculate_offset(slice_array, is_integral);

    // 返回适当维度的span
    if constexpr (NewRank == 0) {
      // 所有维度都是整数索引，返回标量引用
      return data() + offset;
    } else {
      return md::span<T, NewRank, layout_type>(data() + offset, new_extents);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 创建view 视图
  template <typename... Slices>
  auto view(Slices... slices) {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");

    constexpr std::size_t NewRank = compressed_rank_v<Slices...>;

    auto [slice_array, is_integral] = prepare_slices<Rank>(extents(), slices...);

    // 检查越界
    check_slice_bounds<Rank>(slice_array, extents());

    // 计算新的extents
    std::array<std::size_t, NewRank> new_extents;
    std::size_t new_idx = 0;

    for (int i = 0; i < Rank; ++i) {
      if (!is_integral[i]) {  // 只保留非整数索引的维度
        const auto& s = slice_array[i];
        std::ptrdiff_t start = normalize_index(s.start, extent(i));
        std::ptrdiff_t end = normalize_index(s.end, extent(i));
        new_extents[new_idx++] = s.is_all ? extent(i) : 1 + std::floor((end - start) / s.step);
      }
    }

    // 计算步长
    std::array<size_t, NewRank> stride;
    new_idx = 0;
    size_t stride_single = 1;
    size_t last_extent = 1;
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        stride_single = slice_array[i].step * last_extent;
        if (!is_integral[i]) {  // 只保留非整数索引的维度
          stride[NewRank - new_idx++ - 1] = stride_single;
        }
        last_extent *= extent(i);
      }
    } else {
      for (int i = 0; i >= Rank - 1; ++i) {
        stride_single = slice_array[i].step * last_extent;
        if (!is_integral[i]) {  // 只保留非整数索引的维度
          stride[NewRank - new_idx++ - 1] = stride_single;
        }
        last_extent *= extent(i);
      }
    }

    // 计算新的数据指针偏移
    std::size_t offset = calculate_offset(slice_array, is_integral);

    // 返回适当维度的span
    if constexpr (NewRank == 0) {
      // 所有维度都是整数索引，返回标量引用
      return data() + offset;
    } else {
      return md::view<T, NewRank>(data() + offset, new_extents, stride);
    }
  }

 private:
  void check_initialized() const {
    if (mdspan().empty()) {
      throw std::logic_error("md::vector not initialized");
    }
  }
  // 计算数据指针偏移
  std::size_t calculate_offset(const std::array<slice, Rank>& slices, const std::array<bool, Rank>& is_integral) {
    std::size_t offset = 0;
    std::size_t stride = 1;

    // 按内存布局计算偏移（这里以行优先为例）
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        if (!is_integral[i]) {
          offset += slices[i].start * stride;
          stride *= extent(i);
        } else {
          offset += static_cast<std::size_t>(slices[i].start) * stride;
        }
      }
    } else {
      for (int i = 0; i <= Rank - 1; ++i) {
        if (!is_integral[i]) {
          offset += slices[i].start * stride;
          stride *= extent(i);
        } else {
          offset += static_cast<std::size_t>(slices[i].start) * stride;
        }
      }
    }

    return offset;
  }
};

}  // namespace md

#endif  // __MDVECTOR_MD_VECTOR__