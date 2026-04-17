#ifndef __MDARRAY_MD_SPAN__
#define __MDARRAY_MD_SPAN__

#include "../storage/storage.h"
#include "../iterator/iterator.h"
#include "../multi_dim/multi_dim.h"
#include "../expression/expression.h"

namespace md {

template <typename T, size_t Rank, typename Layout = std::layout_right>
class span final : public base_expr<span<T, Rank, Layout>, T>,
                   public view_storage<T>,
                   public multi_dim_dynamic<span<T, Rank, Layout>, T, Rank, Layout>,
                   public iterator_contiguous<span<T, Rank, Layout>, T>,
                   public fill_op<span<T, Rank, Layout>, T>,
                   public expression<span<T, Rank, Layout>, T, unaligned_policy> {
 public:
  using simd_policy = unaligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

  using BaseExpr = base_expr<span<T, Rank, Layout>, T>;
  using Storage = view_storage<T>;
  using MultiDim = multi_dim_dynamic<span<T, Rank, Layout>, T, Rank, Layout>;
  using Iterator = iterator_contiguous<span<T, Rank, Layout>, T>;
  using FillOps = fill_op<span<T, Rank, Layout>, T>;
  using Expr = expression<span<T, Rank, Layout>, T, unaligned_policy>;

  // ============ 构造函数 ============

  span() = default;

  explicit span(T* data, const std::array<size_t, Rank>& shape)
      : Storage(data, calculate_size(shape)), MultiDim(shape, std::make_index_sequence<Rank>{}) {}

  // 删除移动/拷贝 赋值/构造 不管理所有权
  span(const span& other) = delete;

  span(const span&& other) = delete;

  span& operator=(const span& other) = delete;

  span& operator=(span&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~span() = default;

  template <typename E>
  span(const base_expr<E, T>& expr) = delete;

  // ============ 转发接口 ============

  using Storage::data;
  using Storage::size;
  using Storage::used_size;
  using Storage::capacity;
  using Storage::remaining_size;

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
  auto load_simd(size_t i) const noexcept {
    if (i + simd<T2>::pack_size <= size()) {
      return simd_policy::load<T2>(this->data() + i);
    } else {
      return simd_policy::mask_load<T2>(this->data() + i, remaining_size());
    }
  }

  template <typename T2>
  void store_simd(const size_t& i, simd<T2>::const_ref_type val) noexcept {
    if (i + simd<T2>::pack_size <= size()) {
      simd_policy::store<T>(data() + i, val);
    } else {
      simd_policy::mask_store<T>(data() + i, remaining_size(), val);
    }
  }
};

}  // namespace md

#endif  // __MDARRAY_MD_SPAN__