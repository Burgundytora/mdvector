#ifndef __MDVECTOR_SPAN_SUBVIEW_H__
#define __MDVECTOR_SPAN_SUBVIEW_H__

#include "mdspan.h"

// TODO 待完成
// 子视图 不支持simd表达式计算 但支持跨步等python风格切片 支持stl迭代器 基础操作符重载
template <class T, size_t Rank, class Layout = md::layout_right>
class subview {
 public:
  // 默认构造函数
  subview() noexcept = default;

  subview(T* data, const std::array<std::size_t, Rank>& extents, const std::array<md::slice, Rank>& slice_set) {
    md::check_slice_bounds<Rank>(slice_set, extents);
    this->extents_all_ = extents;
  }

 private:
  T* data_ = nullptr;
  size_t size_ = 1;
  std::array<std::size_t, Rank> extents_all_;
  std::array<std::size_t, Rank> extents_view_;
  std::array<std::size_t, Rank> extents_view_to_all_;
  std::array<std::size_t, Rank> strides_;
};

#endif  // MDVECTOR_SPAN_SUBVIEW_H_