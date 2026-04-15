#ifndef __MDVECTOR_DEV_VIEW__
#define __MDVECTOR_DEV_VIEW__

#include "../storage/storage.h"
#include "../iterator/iterator.h"
#include "../multi_dim/multi_dim.h"
#include "../expression/expression.h"

namespace md {

template <typename T, size_t Rank>
class view final : public base_expr<view<T, Rank>, T>,
                   public view_storage<T>,
                   public multi_dim_stride<view<T, Rank>, T, Rank>,
                   //  public iterator_stride<T, Rank>,
                   public fill_ops<view<T, Rank>, T>,
                   public expression<view<T, Rank>, T, aligned_policy> {
 public:
  using simd_policy = aligned_policy;
  using value_type = T;
  using layout_type = std::layout_stride;
  static constexpr size_t rank_ = Rank;

  using BaseExpr = base_expr<view<T, Rank>, T>;
  using Storage = view_storage<T>;
  using MultiDim = multi_dim_stride<view<T, Rank>, T, Rank>;
  // using Iterator = iterator_stride<T, Rank>;
  using FillOps = fill_ops<view<T, Rank>, T>;
  using Expr = expression<view<T, Rank>, T, aligned_policy>;

  // ============ 构造函数 ============

  view() = default;

  explicit view(T* data, const std::array<size_t, Rank>& shape, const std::array<size_t, Rank>& stride)
      : Storage(data, calculate_size(shape)) {
    MultiDim::init_mdspan(shape, stride);
  }

  // 删除移动/拷贝 赋值/构造 不管理所有权
  view(const view& other) = delete;

  view(const view&& other) = delete;

  view& operator=(const view& other) = delete;

  view& operator=(view&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~view() = default;

  template <typename E>
  view(const base_expr<E, T>& expr) = delete;

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

  // using Iterator::begin;
  // using Iterator::end;
  // using Iterator::cbegin;
  // using Iterator::cend;
  // using Iterator::rbegin;
  // using Iterator::rend;
  // using Iterator::crbegin;
  // using Iterator::crend;
  // 迭代器类型定义
  using iterator = iterator_stride<T, Rank, false>;
  using const_iterator = iterator_stride<T, Rank, true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() noexcept { return iterator(this, 0); }
  iterator end() noexcept { return iterator(this, size()); }
  const_iterator begin() const noexcept { return const_iterator(this, 0); }
  const_iterator end() const noexcept { return const_iterator(this, size()); }
  const_iterator cbegin() const noexcept { return const_iterator(this, 0); }
  const_iterator cend() const noexcept { return const_iterator(this, size()); }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

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
  typename simd<T2>::type load_simd(size_t i) const noexcept {
    // 内存不连续 使用对齐的buffer转存
    if (i + simd<T2>::pack_size <= size()) {
      alignas(simd<T>::alignment) T buffer[simd<T>::pack_size];
      size_t temp_i = i;
      for (size_t j = 0; j < simd<T>::pack_size && temp_i < used_size(); ++j, ++temp_i) {
        buffer[j] = *const_iterator(this, temp_i);  // 使用 const_iterator
      }
      return simd_policy::template load<T2>(buffer);
    } else {
      alignas(simd<T>::alignment) T buffer[simd<T>::pack_size];
      size_t temp_i = i;
      size_t count = 0;
      for (; count < simd<T>::pack_size && temp_i < used_size(); ++count, ++temp_i) {
        buffer[count] = *const_iterator(this, temp_i);  // 使用 const_iterator
      }
      return simd_policy::template mask_load<T2>(buffer, remaining_size());
    }
  }

  template <typename T2>
  void store_simd(size_t i, typename simd<T2>::const_ref_type val) noexcept {
    // 先将simd转换为普通变量再用迭代器赋值
    if (i + simd<T2>::pack_size <= size()) {
      alignas(simd<T>::alignment) T buffer[simd<T>::pack_size];
      simd_policy::template store<T>(buffer, val);
      for (size_t j = 0; j < simd<T>::pack_size && (i + j) < used_size(); ++j) {
        *iterator(this, i + j) = buffer[j];
      }
    } else {
      alignas(simd<T>::alignment) T buffer[simd<T>::pack_size];
      simd_policy::template mask_store<T>(buffer, remaining_size(), val);
      for (size_t j = 0; j < remaining_size() && (i + j) < used_size(); ++j) {
        *iterator(this, i + j) = buffer[j];
      }
    }
  }
};

}  // namespace md

#endif  // __MDVECTOR_DEV_VIEW__