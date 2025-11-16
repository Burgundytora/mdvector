#ifndef __MDVECTOR_VIEW_H__
#define __MDVECTOR_VIEW_H__

#include "common/iterator_view.h"
#include "expression_template/operator.h"
#include "simd/simd_function.h"

namespace md {

// 带步长功能 不要求内存连续视图 基于std::layout_stride
template <typename T, size_t Rank>
class view : public md::tensor_expr<view<T, Rank>, T> {
 public:
  using Policy = md::unaligned_policy;
  using value_type = T;
  using layout_type = std::layout_stride;
  static constexpr size_t rank_ = Rank;

  // 迭代器类型定义
  using iterator = view_iterator<T, Rank, false>;
  using const_iterator = view_iterator<T, Rank, true>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 protected:
  std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride> mdspan_;
  std::array<size_t, Rank> shape_;
  size_t size_;

 public:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 构造函数
  constexpr view() noexcept = default;

  view(T* data, const std::array<std::size_t, Rank>& shape, const std::array<std::size_t, Rank>& stride)
      : mdspan_(create_mdspan(data, shape, stride, std::make_index_sequence<Rank>{})),
        shape_(shape),
        size_(md::calculate_size(shape)) {}

  view(const view& other) = delete;

  view(const view&& other) = delete;

  view& operator=(const view& other) = delete;

  // 删除移动赋值 不管理所有权
  view& operator=(view&& other) = delete;

  // 析构使用自动生成 不会销毁指针数组
  ~view() = default;

  template <typename E>
  view(const md::tensor_expr<E, T>& expr) = delete;

  // TODO
  // template <typename E>
  // view& operator=(const md::tensor_expr<E, T>& expr) noexcept {
  //   expr.template eval_to<T, Policy>(this->data());
  //   return *this;
  // }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 访问属性
  T* data() { return mdspan_.data_handle(); }

  const T* data() const { return mdspan_.data_handle(); }

  size_t used_size() const noexcept { return size_; }

  size_t size() const noexcept { return size_; }

  void fill(T val) { std::fill(begin(), end(), val); }

  auto extents() const { return shape_; }

  size_t extent(int index) const { return shape_.at(index); }

  bool empty() { return mdspan_.empty(); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 多维索引
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& operator()(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& operator[](Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    return mdspan_[indices...];
  }

  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  template <typename... Indices>
  const T& at(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    check_indices(indices...);
    return mdspan_[indices...];
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 索引转换
  // 1d_index为ptr距离
  template <typename... Indices>
  size_t get_1d_index(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match rank");
    // 使用 mdspan 的 mapping 来获取线性索引
    return mdspan_.mapping()(indices...);
  }

  // md_index为mdspan中索引不带步长信息
  // 一维索引转多维索引
  std::array<size_t, Rank> get_md_index(size_t linear_index) const {
    if (linear_index >= size_) {
      throw std::out_of_range("Linear index out of range");
    }
    std::array<size_t, Rank> indices{};
    size_t remaining = linear_index;
    for (int i = Rank - 1; i >= 0; --i) {
      indices[i] = remaining % shape_[i];
      remaining /= shape_[i];
    }
    return indices;
  }

  // 获取指定维度的索引
  size_t get_dim_index(size_t linear_index, size_t dim) const {
    if (linear_index >= size_ || dim >= Rank) {
      throw std::out_of_range("Index out of range");
    }

    return get_md_index(linear_index)[dim];
  }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 迭代器实现
  iterator begin() noexcept { return iterator(this, 0); }

  iterator end() noexcept { return iterator(this, size_); }

  const_iterator begin() const noexcept { return const_iterator(this, 0); }

  const_iterator end() const noexcept { return const_iterator(this, size_); }

  const_iterator cbegin() const noexcept { return const_iterator(this, 0); }

  const_iterator cend() const noexcept { return const_iterator(this, size_); }

  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

  ///////////////////////////////////////////////////////////////////////////////////////
  /// 打印
  void print()
    requires Printable<T>
  {
    if (!mdspan_.empty()) {
      md::print_mdspan(mdspan_);
    } else {
      throw std::logic_error("md::span need to be initialized before print!!!");
    }
  }

 private:
  ///////////////////////////////////////////////////////////////////////////////////////
  /// 内部函数
  template <size_t... Is>
  auto create_mdspan(T* data, const std::array<size_t, Rank>& shape, const std::array<std::size_t, Rank>& stride,
                     std::index_sequence<Is...>) {
    return std::mdspan<T, std::dextents<size_t, Rank>, std::layout_stride>(
        data,
        std::layout_stride::mapping<std::dextents<size_t, Rank>>(std::dextents<size_t, Rank>(shape[Is]...), stride));
  }

  template <typename... Indices>
  void check_indices(Indices... indices) const {
    static_assert(sizeof...(Indices) == Rank, "Number of indices must match the rank of mdvector");

    const size_t idx_array[Rank] = {static_cast<size_t>(indices)...};
    for (int i = 0; i < Rank; ++i) {
      if (idx_array[i] >= mdspan_.extent(i)) {
        throw std::out_of_range(
            std::format("Index {} out of range for dimension {} (size: {})", idx_array[i], i, mdspan_.extent(i)));
      }
    }
  }
};

}  // namespace md

#endif  //__MDVECTOR_VIEW_H__