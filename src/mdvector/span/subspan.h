#ifndef MDVECTOR_SPAN_SUBSPAN_H_
#define MDVECTOR_SPAN_SUBSPAN_H_

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "mdspan.h"

template <class T, size_t Rank, class Layout = layout_right>
class subspan : public mdspan<T, Rank, Layout> {
 public:
  subspan() = delete;

  subspan(T* data, const std::array<std::size_t, Rank>& extents, const std::array<detail::Slice, Rank>& slice_set)
      : mdspan<T, Rank, Layout>(nullptr, {}) {
    check_slice_bounds(slice_set, extents);
    if (!is_contiguous_slice(slice_set)) {
      throw std::runtime_error("subspan slices must result in contiguous memory");
    }

    std::array<std::size_t, Rank> new_extents;
    std::array<std::size_t, Rank> new_strides = detail::compute_strides(extents);
    std::size_t offset = 0;

    for (size_t i = 0; i < Rank; ++i) {
      if (slice_set[i].is_all) {
        new_extents[i] = extents[i];
      } else {
        // 修改为闭区间计算方式 [start, end] → size = end - start + 1
        new_extents[i] = slice_set[i].end - slice_set[i].start + 1;
        offset += slice_set[i].start * new_strides[i];
      }
    }

    this->data_ = data + offset;
    this->extents_ = new_extents;
    this->strides_ = new_strides;
  }

  bool is_contiguous() const { return true; }

 private:
  void check_slice_bounds(const std::array<detail::Slice, Rank>& slices, const std::array<std::size_t, Rank>& extents) {
    for (size_t i = 0; i < Rank; ++i) {
      if (slices[i].start < 0 || slices[i].end < 0) {
        throw std::out_of_range("subspan slice indices cannot be negative");
      }

      if (!slices[i].is_all) {
        // 修改边界检查逻辑为闭区间
        if (static_cast<std::size_t>(slices[i].start) >= extents[i] ||
            static_cast<std::size_t>(slices[i].end) >= extents[i]) {
          throw std::out_of_range("subspan slice out of range");
        }
        if (slices[i].start > slices[i].end) {  // 允许start == end（单元素）
          throw std::invalid_argument("subspan slice start must <= end");
        }
      }
    }
  }

  bool is_contiguous_slice(const std::array<detail::Slice, Rank>& slices) {
    for (int i = Rank - 1; i >= 0; --i) {
      if (!slices[i].is_all) {
        for (int j = i - 1; j >= 0; j--) {
          // 单元素检查改为 start == end
          if (slices[j].start != slices[j].end) {
            return false;
          }
        }
        break;
      }
    }
    return true;
  }
};

#endif  // MDVECTOR_SPAN_SUBSPAN_H_