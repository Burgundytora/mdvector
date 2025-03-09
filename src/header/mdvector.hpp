#ifndef HEADER_MDVECTOR_HPP_
#define HEADER_MDVECTOR_HPP_

#include <mdspan>
#include <array>

#include "allocator.h"

// 表达式模板基类
template <typename Derived>
class Expr {
 public:
  auto operator[](size_t i) const { return static_cast<const Derived&>(*this)[i]; }
  size_t size() const { return static_cast<const Derived&>(*this).size(); }
};

// 核心MDVector类
template <typename T, size_t Dims>
class MDVector : public Expr<MDVector<T, Dims>> {
 public:
  // mdspan类型别名定义
  using extents_type = std::dextents<size_t, Dims>;
  using layout_type = std::layout_right;
  using mdspan_type = std::mdspan<T, extents_type, layout_type>;

 private:
  AlignedAllocator<T> allocator_;
  T* data_;
  mdspan_type view_;
  std::array<size_t, Dims> dimensions_;
  size_t total_elements_;
  size_t aligned_size_;

 public:
  // 构造函数
  template <typename... len>
  MDVector(len... dims) : dimensions_{static_cast<size_t>(dims)...} {
    static_assert(sizeof...(dims) == Dims, "维度数量不匹配");
    total_elements_ = 1;
    for (auto d : dimensions_) total_elements_ *= d;
    data_ = allocator_.allocate(total_elements_);
    view_ = mdspan_type(data_, dimensions_);
  }

  // ========================================================
  // 访问运算符 提供safe 和 unsafe两种方式
  // (i, j, k) unsafe style
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "索引数量与维度不匹配");
    return view_[static_cast<size_t>(indices)...];
  }

  // [i, j, k] unsafe style
  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "索引数量与维度不匹配");
    return view_[static_cast<size_t>(indices)...];
  }

  // .at(i, j, k) safe style
  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "Number of indices must match dimensions");
    size_t i = 0;
    for (auto len : std::array<size_t, Dims>{static_cast<size_t>(indices)...}) {
      if (len > view_.extent(i)) {
        std::cerr << "mdspan out-of-range error: " << len << ">" << view_.extent(i) << "\n";
        std::abort();
      }
      i++;
    }
    return view_[indices...];
  }
  // ========================================================

  T* data() const { return data_; }
  size_t size() const { return total_elements_; }

 private:
};

#endif  // HEADER_MDVECTOR_HPP_
