#pragma once

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
  using FillOp = fill_op<vector<T, Rank, Layout>, T>;
  using Expr = expression<vector<T, Rank, Layout>, T, aligned_policy>;

  // ============ 构造函数 ============

  vector() = default;

  explicit vector(const std::array<size_t, Rank>& shape)
      : Storage(calculate_size(shape)), MultiDim(shape, std::make_index_sequence<Rank>{}) {}

  template <typename... Sizes>
    requires(sizeof...(Sizes) == Rank && (std::convertible_to<Sizes, size_t> && ...))
  explicit vector(Sizes... sizes) : vector(std::array<size_t, Rank>{static_cast<size_t>(sizes)...}) {}

  // 从表达式构造
  template <typename E, typename U>
  vector(const base_expr<E, U>& expr)
    requires(Numeric<T> && std::convertible_to<U, T>)
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

  template <typename E, typename U>
  vector& operator=(const base_expr<E, U>& expr)
    requires(Numeric<T> && std::convertible_to<U, T>)
  {
    set_shape(expr.extents());
    expr.template eval_to<>(*this);
    return *this;
  }

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

  using FillOp::fill;
  using FillOp::set_zeros;
  using FillOp::set_ones;
  using FillOp::set_arange;
  using FillOp::set_random_uniform;
  using FillOp::set_random_normal;

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
    std::ptrdiff_t offset = calculate_offset(slice_array, is_integral);

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
        if (s.is_all) new_extents[new_idx++] = extent(i);
        else if ((s.step > 0 && start > end) || (s.step < 0 && start < end)) new_extents[new_idx++] = 0;
        else new_extents[new_idx++] = static_cast<size_t>(1 + (s.step > 0 ? (end - start) / s.step : (start - end) / (-s.step)));
      }
    }

    // 计算步长
    std::array<std::ptrdiff_t, NewRank> stride;
    new_idx = 0;
    std::ptrdiff_t stride_single = 1;
    std::ptrdiff_t last_extent = 1;
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        stride_single = slice_array[i].step * last_extent;
        if (!is_integral[i]) {  // 只保留非整数索引的维度
          stride[NewRank - new_idx++ - 1] = stride_single;
        }
        last_extent *= extent(i);
      }
    } else {
      new_idx = 0;
      for (int i = 0; i <= Rank - 1; ++i) {
        stride_single = slice_array[i].step * last_extent;
        if (!is_integral[i]) {  // 只保留非整数索引的维度
          stride[new_idx++] = stride_single;
        }
        last_extent *= extent(i);
      }
    }

    // 计算新的数据指针偏移
    std::ptrdiff_t offset = calculate_offset(slice_array, is_integral);

    // 返回适当维度的span
    if constexpr (NewRank == 0) {
      // 所有维度都是整数索引，返回标量引用
      return data() + offset;
    } else {
      return md::view<T, NewRank>(data() + offset, new_extents, stride);
    }
  }

  // Read-only view overload.  The returned view carries const T in its type,
  // so mutation and expression assignment are rejected at compile time.
  template <typename... Slices>
  auto view(Slices... slices) const {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");
    constexpr std::size_t NewRank = compressed_rank_v<Slices...>;
    auto [slice_array, is_integral] = prepare_slices<Rank>(extents(), slices...);
    check_slice_bounds<Rank>(slice_array, extents());
    std::array<std::size_t, NewRank> new_extents{};
    std::size_t new_idx = 0;
    for (int i = 0; i < Rank; ++i) if (!is_integral[i]) {
      const auto& s = slice_array[i];
      const std::ptrdiff_t start = normalize_index(s.start, extent(i));
      const std::ptrdiff_t end = normalize_index(s.end, extent(i));
      new_extents[new_idx++] = s.is_all ? extent(i) :
          ((s.step > 0 && start > end) || (s.step < 0 && start < end) ? 0 :
           static_cast<size_t>(1 + (s.step > 0 ? (end - start) / s.step : (start - end) / (-s.step))));
    }
    std::array<std::ptrdiff_t, NewRank> stride{};
    new_idx = 0;
    std::ptrdiff_t stride_single = 1, last_extent = 1;
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        stride_single = slice_array[i].step * last_extent;
        if (!is_integral[i]) stride[NewRank - new_idx++ - 1] = stride_single;
        last_extent *= static_cast<std::ptrdiff_t>(extent(i));
      }
    } else {
      new_idx = 0;
      for (int i = 0; i < Rank; ++i) {
        stride_single = slice_array[i].step * last_extent;
        if (!is_integral[i]) stride[new_idx++] = stride_single;
        last_extent *= static_cast<std::ptrdiff_t>(extent(i));
      }
    }
    std::ptrdiff_t offset = 0, base_stride = 1;
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) { offset += slice_array[i].start * base_stride; base_stride *= extent(i); }
    } else {
      for (int i = 0; i < Rank; ++i) { offset += slice_array[i].start * base_stride; base_stride *= extent(i); }
    }
    if constexpr (NewRank == 0) return static_cast<const T*>(data()) + offset;
    else return md::view<const T, NewRank>(data() + offset, new_extents, stride);
  }

 private:
  void check_initialized() const {
    if (mdspan().empty()) {
      throw std::logic_error("md::vector not initialized");
    }
  }
  // 计算数据指针偏移
  std::ptrdiff_t calculate_offset(const std::array<slice, Rank>& slices, const std::array<bool, Rank>& is_integral) {
    std::ptrdiff_t offset = 0;
    std::ptrdiff_t stride = 1;

    // 按内存布局计算偏移（这里以行优先为例）
    if constexpr (std::is_same_v<Layout, std::layout_right>) {
      for (int i = Rank - 1; i >= 0; --i) {
        if (!is_integral[i]) {
          offset += slices[i].start * stride;
          stride *= extent(i);
        } else {
          offset += slices[i].start * stride;
        }
      }
    } else {
      for (int i = 0; i <= Rank - 1; ++i) {
        if (!is_integral[i]) {
          offset += slices[i].start * stride;
          stride *= extent(i);
        } else {
          offset += slices[i].start * stride;
        }
      }
    }

    return offset;
  }
};

// 编译期检查multi_dim_dynamic
namespace detail {

template <typename T, size_t Rank = 2, typename Layout = std::layout_right>
struct multi_dim_dynamci_checks {
  using test_mdd = multi_dim_dynamic<vector<T, Rank, Layout>, T, Rank, Layout>;
  static_assert(BasicMultiDim<test_mdd, T, Rank>);
  static_assert(MultiDimIndexable<test_mdd, T, Rank, size_t, size_t>);
  static_assert(MultiDimBoundsCheck<test_mdd, T, Rank, size_t, size_t>);
  static_assert(MultiDimIndexConvert<test_mdd, Rank, size_t, size_t>);
  static_assert(MultiDimShapeMutable<test_mdd, Rank>);
};

template struct multi_dim_dynamci_checks<float>;
template struct multi_dim_dynamci_checks<double>;
template struct multi_dim_dynamci_checks<int>;

}  // namespace detail

}  // namespace md
