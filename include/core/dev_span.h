#ifndef __MDVECTOR_DEV_SPAN__
#define __MDVECTOR_DEV_SPAN__

#include "../storage/storage.h"
#include "../iterator/iterator.h"
#include "../multi_dim/multi_dim.h"
#include "../expression/expression.h"

namespace md {

template <typename T, size_t Rank, typename Layout = std::layout_right>
class span : public base_expr<span<T, Rank, Layout>, T>,
             public span_storage<T>,
             public multi_dim_dynamic<span<T, Rank, Layout>, T, Rank, Layout>,
             public iterator_contiguous<span<T, Rank, Layout>, T>,
             public fill_ops<span<T, Rank, Layout>, T>,
             public expression_impl<span<T, Rank, Layout>, T, unaligned_policy> {
  using BaseExpr = base_expr<span<T, Rank, Layout>, T>;
  using Storage = span_storage<T>;
  using MultiDim = multi_dim_dynamic<span<T, Rank, Layout>, T, Rank, Layout>;
  using Iterator = iterator_contiguous<span<T, Rank, Layout>, T>;
  using FillOps = fill_ops<span<T, Rank, Layout>, T>;
  using Expr = expression_impl<span<T, Rank, Layout>, T, unaligned_policy>;

  friend Storage;
  friend MultiDim;
  friend Expr;

 public:
  using simd_policy = unaligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

  // ============ 构造函数 ============

  span() = default;

  explicit span(T* data, const std::array<size_t, Rank>& shape) : Storage(data, calculate_size(shape)) {
    MultiDim::shape_ = shape;
    MultiDim::init_mdspan();
  }

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
  using Storage::remaining_size;  // span特有

  using FillOps::fill;
  using FillOps::set_zeros;
  using FillOps::set_ones;
  using FillOps::set_arange;
  using FillOps::set_random_uniform;
  using FillOps::set_random_normal;

  using MultiDim::extents;
  using MultiDim::extent;
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

  using Expr::operator=;
  using Expr::operator+=;
  using Expr::operator-=;
  using Expr::operator*=;
  using Expr::operator/=;
  using Expr::operator-;
  using Expr::operator+;

  // ============ SIMD 接口 ============

  template <typename T2>
  auto load_simd(size_t i) const noexcept {
    if (i + simd<T2>::pack_size <= size_) {
      return Policy::load<T2>(this->data() + i);
    } else {
      return Policy::mask_load<T2>(this->data() + i, remaining_size());
    }
  }

  template <typename T2>
  void store_simd(const size_t& i, simd<T2>::const_ref_type simd_val) noexcept {
    if (i + simd<T2>::pack_size <= size_) {
      Policy::store<T>(this->data() + i, simd_val);
    } else {
      Policy::mask_store<T>(this->data() + i, remaining_size(), simd_val);
    }
  }
};

}  // namespace md

#endif  // __MDVECTOR_DEV_SPAN__