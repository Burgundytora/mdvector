#ifndef MDVECTOR_SPAN_SUBSPAN_H_
#define MDVECTOR_SPAN_SUBSPAN_H_

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "mdspan.h"

template <class T, size_t Rank, class Layout = layout_right>
class subspan : public mdspan<T, Rank, Layout>, public Expr<subspan<T, Rank, Layout>> {
 public:
  // 默认构造函数
  constexpr subspan() noexcept = default;

  // 详细构造函数
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

    size_ = 1;
    for (auto s : extents) {
      size_ *= s;
    }
  }

  // 允许拷贝构造
  subspan(const subspan& other) = default;

  // 禁止移动构造
  subspan(const subspan&& other) = delete;

  // ---------- 赋值运算符 ----------
  // 拷贝赋值：将右侧数据复制到当前视图（不修改指针和大小）
  subspan& operator=(const subspan& other) {
    std::copy(other.begin(), other.end(), data_);  // 逐元素复制
    return *this;
  }

  // 删除移动赋值 不管理所有权
  subspan& operator=(subspan&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~subspan() = default;

  // ====================== 迭代器 ============================

  T* begin() { return data_; }
  T* end() { return data_ + size_; }  // 尾后指针

  // const 重载
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }

  // ====================== 数值运算 ============================

  void set_value(T val) { std::fill(begin(), end(), val); }

  // =================== 表达式模板 ============================
  // 不允许表达式构造

  // 表达式赋值
  template <typename E>
  subspan& operator=(const Expr<E>& expr) {
    expr.eval_to(this->data());  // 直接计算到目标内存
    return *this;
  }

  // 取值
  template <typename T2>
  typename simd<T2>::type eval_simd(size_t i) const {
    return simd<T2>::load(data() + i);
  }

  // 取值
  template <typename T2>
  typename simd<T2>::type eval_simd_mask(size_t i) const {
    return simd<T2>::mask_load(data() + i, size() - i);
  }

  // ========================================================
  // b ?= a
  subspan& operator+=(const subspan& other) {
    simd_add_inplace(this->data(), other.data(), this->size());
    return *this;
  }

  subspan& operator-=(const subspan& other) {
    simd_sub_inplace(this->data(), other.data(), this->size());
    return *this;
  }

  subspan& operator*=(const subspan& other) {
    simd_mul_inplace(this->data(), other.data(), this->size());
    return *this;
  }

  subspan& operator/=(const subspan& other) {
    simd_div_inplace(this->data(), other.data(), this->size());
    return *this;
  }

  // ========================================================

  // 标量操作
  // 添加标量eval_scalar方法
  template <typename T2>
  T2 eval_scalar(size_t i) const {
    return static_cast<T2>(data_[i]);
  }

  // 添加与标量的复合赋值运算符
  subspan& operator+=(T scalar) {
    simd_add_inplace_scalar(this->data(), scalar, this->size());
    return *this;
  }

  subspan& operator-=(T scalar) {
    simd_sub_inplace_scalar(this->data(), scalar, this->size());
    return *this;
  }

  subspan& operator*=(T scalar) {
    simd_mul_inplace_scalar(this->data(), scalar, this->size());
    return *this;
  }

  subspan& operator/=(T scalar) {
    simd_div_inplace_scalar(this->data(), scalar, this->size());
    return *this;
  }

  // ====================== 标量运算 ============================

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