// core/dev_mdvector.h
#ifndef __MDVECTOR_DEV_MDVECTOR__
#define __MDVECTOR_DEV_MDVECTOR__

#include "storage/storage_data.h"
#include "fill.h"
#include "multi_dim.h"
#include "expression.h"
#include "../common/iterator_mixin.h"
#include "../expression_template/operator_overload.h"

namespace md {

template <typename T, size_t Rank, typename Layout = std::layout_right>
class vector : public base_expr<vector<T, Rank, Layout>, T>,
               public vector_storage<T>,
               public iterator_mixin<vector<T, Rank, Layout>, T>,
               public multi_dim_impl<vector<T, Rank, Layout>, T, Rank, Layout>,
               public fill_ops<vector<T, Rank, Layout>, T>,
               public expression_impl<vector<T, Rank, Layout>, T, aligned_policy> {
  using BaseExpr = base_expr<vector<T, Rank, Layout>, T>;
  using Storage = vector_storage<T>;
  using FillOps = fill_ops<vector<T, Rank, Layout>, T>;
  using MultiDim = multi_dim_impl<vector<T, Rank, Layout>, T, Rank, Layout>;
  using Expr = expression_impl<vector<T, Rank, Layout>, T, aligned_policy>;
  using Iterator = iterator_mixin<vector<T, Rank, Layout>, T>;

  friend Storage;
  friend MultiDim;
  friend Expr;

 public:
  using simd_policy = aligned_policy;
  using value_type = T;
  using layout_type = Layout;
  static constexpr size_t rank_ = Rank;

  // ============ 构造函数 ============

  vector() = default;

  explicit vector(const std::array<size_t, Rank>& shape) : Storage(calculate_size(shape)) {
    MultiDim::shape_ = shape;
    MultiDim::init_mdspan();
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
    MultiDim::init_mdspan();
    expr.template eval_to<>(*this);
  }

  // 拷贝/移动
  vector(const vector& other) : Storage(other.size_), MultiDim() {
    MultiDim::shape_ = other.shape_;
    MultiDim::init_mdspan();
    std::copy(other.begin(), other.end(), this->begin());
  }

  vector(vector&& other) noexcept = default;

  vector& operator=(const vector& other) {
    if (this != &other) {
      Storage::resize(other.size_);
      MultiDim::shape_ = other.shape_;
      MultiDim::init_mdspan();
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
    MultiDim::init_mdspan();
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
  auto load_simd(size_t i) const noexcept
    requires Numeric<T>
  {
    return simd_policy::template load<T2>(data() + i);
  }

  template <typename T2>
  void store_simd(size_t i, typename simd<T2>::const_ref_type val) noexcept {
    simd_policy::template store<T>(data() + i, val);
  }
};

// 便捷别名
template <typename T>
using vector_1d = vector<T, 1>;
template <typename T>
using vector_2d = vector<T, 2>;
template <typename T>
using vector_3d = vector<T, 3>;

}  // namespace md

#endif