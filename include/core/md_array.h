#pragma once

#include "../storage/storage.h"
#include "../iterator/iterator.h"
#include "../multi_dim/multi_dim.h"
#include "../expression/expression.h"

namespace md {

template <typename T, typename Layout = std::layout_right, size_t... lengths>
class array final : public base_expr<array<T, Layout, lengths...>, T>,
                    public stack_storage<T, lengths...>,
                    public multi_dim_static<array<T, Layout, lengths...>, T, Layout, lengths...>,
                    public iterator_contiguous<array<T, Layout, lengths...>, T>,
                    public fill_op<array<T, Layout, lengths...>, T>,
                    public expression<array<T, Layout, lengths...>, T, aligned_policy> {
 public:
  using simd_policy = aligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = sizeof...(lengths);

  using BaseExpr = base_expr<array<T, Layout, lengths...>, T>;
  using Storage = stack_storage<T, lengths...>;
  using MultiDim = multi_dim_static<array<T, Layout, lengths...>, T, Layout, lengths...>;
  using Iterator = iterator_contiguous<array<T, Layout, lengths...>, T>;
  using FillOp = fill_op<array<T, Layout, lengths...>, T>;
  using Expr = expression<array<T, Layout, lengths...>, T, aligned_policy>;

  // ============ 构造函数 ============

  array() = default;

  ~array() = default;

  array(const array& other) = default;

  array(array&& other) = default;

  array& operator=(const array& other) = default;

  array& operator=(array&& other) = default;

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
};

// 编译期检查multi_dim_static
namespace detail {

template <typename T, size_t Rank = 2>
struct multi_dim_static_checks {
  using Layout = std::layout_right;
  using test_mds = multi_dim_static<array<T, Layout, 2, 2>, T, Layout, 2, 2>;
  static_assert(BasicMultiDim<test_mds, T, Rank>);
  static_assert(MultiDimIndexable<test_mds, T, Rank, size_t, size_t>);
  static_assert(MultiDimBoundsCheck<test_mds, T, Rank, size_t, size_t>);
  static_assert(MultiDimIndexConvert<test_mds, Rank, size_t, size_t>);
  static_assert(MultiDimStatic<test_mds, Rank>);
};

template struct multi_dim_static_checks<float>;
template struct multi_dim_static_checks<double>;
template struct multi_dim_static_checks<int>;

}  // namespace detail

}  // namespace md
