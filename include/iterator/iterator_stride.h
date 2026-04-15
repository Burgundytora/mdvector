#ifndef __MDVECTOR_INTERAOTR_STRIDE__
#define __MDVECTOR_INTERAOTR_STRIDE__

#include "../concepts/base_concept.h"

namespace md {

// 前向声明
template <typename T, size_t Rank>
class view;

// TODO: 优化性能
template <typename T, size_t Rank, bool IsConst>
class iterator_stride {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = std::conditional_t<IsConst, const T, T>;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;
  using view_type = std::conditional_t<IsConst, const view<T, Rank>, view<T, Rank>>;

 private:
  view_type* view_ptr_;
  std::array<size_t, Rank> current_indices_;
  size_t linear_pos_;

 public:
  // 构造函数
  iterator_stride(view_type* view, std::array<size_t, Rank> indices, size_t linear_pos)
      : view_ptr_(view), current_indices_(indices), linear_pos_(linear_pos) {}

  iterator_stride(view_type* view, size_t linear_pos = 0) : view_ptr_(view), linear_pos_(linear_pos) {
    if (view_ptr_ && linear_pos_ < view_ptr_->size()) {
      current_indices_ = view_ptr_->get_md_index(linear_pos_);
    } else {
      current_indices_.fill(0);
    }
  }

  // 允许从非 const 迭代器构造 const 迭代器
  template <bool OtherIsConst, typename = std::enable_if_t<IsConst && !OtherIsConst>>
  iterator_stride(const iterator_stride<T, Rank, OtherIsConst>& other)
      : view_ptr_(other.view_ptr_), current_indices_(other.current_indices_), linear_pos_(other.linear_pos_) {}

  // 解引用
  reference operator*() const {
    return std::apply([this](auto... indices) -> reference { return (*view_ptr_)(indices...); }, current_indices_);
  }

  pointer operator->() const { return &(**this); }

  // 前缀递增
  iterator_stride& operator++() {
    increment();
    return *this;
  }

  // 后缀递增
  iterator_stride operator++(int) {
    iterator_stride tmp = *this;
    increment();
    return tmp;
  }

  // 前缀递减
  iterator_stride& operator--() {
    decrement();
    return *this;
  }

  // 后缀递减
  iterator_stride operator--(int) {
    iterator_stride tmp = *this;
    decrement();
    return tmp;
  }

  // 算术运算
  iterator_stride& operator+=(difference_type n) {
    if (n >= 0) {
      for (difference_type i = 0; i < n; ++i) {
        increment();
      }
    } else {
      for (difference_type i = 0; i < -n; ++i) {
        decrement();
      }
    }
    return *this;
  }

  iterator_stride& operator-=(difference_type n) { return *this += (-n); }

  iterator_stride operator+(difference_type n) const {
    iterator_stride tmp = *this;
    return tmp += n;
  }

  iterator_stride operator-(difference_type n) const {
    iterator_stride tmp = *this;
    return tmp -= n;
  }

  difference_type operator-(const iterator_stride& other) const {
    return static_cast<difference_type>(linear_pos_) - static_cast<difference_type>(other.linear_pos_);
  }

  // 下标访问
  reference operator[](difference_type n) const { return *(*this + n); }

  // 比较运算符
  bool operator==(const iterator_stride& other) const {
    return view_ptr_ == other.view_ptr_ && linear_pos_ == other.linear_pos_;
  }

  bool operator!=(const iterator_stride& other) const { return !(*this == other); }

  bool operator<(const iterator_stride& other) const { return linear_pos_ < other.linear_pos_; }

  bool operator<=(const iterator_stride& other) const { return linear_pos_ <= other.linear_pos_; }

  bool operator>(const iterator_stride& other) const { return linear_pos_ > other.linear_pos_; }

  bool operator>=(const iterator_stride& other) const { return linear_pos_ >= other.linear_pos_; }

  // 获取当前位置
  size_t linear_position() const { return linear_pos_; }
  const std::array<size_t, Rank>& indices() const { return current_indices_; }

  // 获取底层视图指针（用于调试）
  view_type* get_view_ptr() const { return view_ptr_; }

 private:
  void increment() {
    if (!view_ptr_ || linear_pos_ >= view_ptr_->size()) {
      return;
    }

    linear_pos_++;
    if (linear_pos_ < view_ptr_->size()) {
      current_indices_ = view_ptr_->get_md_index(linear_pos_);
    }
  }

  void decrement() {
    if (!view_ptr_ || linear_pos_ == 0) {
      return;
    }

    linear_pos_--;
    current_indices_ = view_ptr_->get_md_index(linear_pos_);
  }
};

}  // namespace md

#endif  // __MDVECTOR_INTERAOTR_STRIDE__