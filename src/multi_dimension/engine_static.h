#ifndef __MDVECTOR_MDENGINE_STATIC_H__
#define __MDVECTOR_MDENGINE_STATIC_H__

#include <array>
#include <iostream>
#include <numeric>
#include <string>

#include "expression_template/operator.h"
#include "mdspan.h"
#include "simd/simd_function.h"

namespace md {

template <class T, class Layout = layout_right, size_t... lengths>
class EngineStatic {
 protected:
  // 计算原始总大小
  static constexpr size_t raw_total_size = (lengths * ... * 1);

  // 栈上元素数量对齐到simd 提高性能
  static constexpr size_t total_size = (raw_total_size % simd<T>::pack_size == 0)
                                           ? raw_total_size
                                           : ((raw_total_size / simd<T>::pack_size) + 1) * simd<T>::pack_size;
  alignas(simd<T>::alignment) std::array<T, total_size> data_;
  mdspan<T, sizeof...(lengths), Layout> view_;

 public:
  explicit EngineStatic()
      : view_(mdspan<T, sizeof...(lengths), Layout>(
            data_.data(), std::array<std::size_t, sizeof...(lengths)>{static_cast<std::size_t>(lengths)...})) {
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "T must be trivial and standard-layout!");
    data_.fill(1.0);  // 防止幽灵元素除0
  }

  explicit EngineStatic(T val)
      : view_(mdspan<T, sizeof...(lengths), Layout>(
            data_.data(), std::array<std::size_t, sizeof...(lengths)>{static_cast<std::size_t>(lengths)...})) {
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "T must be trivial and standard-layout!");
    data_.fill(val);
  }

  ~EngineStatic() = default;

  EngineStatic(const EngineStatic& other)
      : data_(other.data_), view_(data_.data(), other.view_.extents()) {}  // 初始化列表

  EngineStatic(EngineStatic&& other) noexcept
      : data_(std::move(other.data_)), view_(data_.data(), other.view_.extents()) {}

  EngineStatic& operator=(const EngineStatic& other) {
    if (this != &other) {
      data_ = other.data_;                                                                 // 复制数据
      view_ = mdspan<T, sizeof...(lengths), Layout>(data_.data(), other.view_.extents());  // 视图重新创建
    }
    return *this;
  }

  EngineStatic& operator=(EngineStatic&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      view_ = mdspan<T, sizeof...(lengths), Layout>(data_.data(), other.view_.extents());
    }
    return *this;
  }

  template <class... Indices>
  T& operator()(Indices... indices) {
    return view_(indices...);
  }

  template <class... Indices>
  const T& operator()(Indices... indices) const {
    return view_(indices...);
  }

  template <class... Indices>
  T& at(Indices... indices) {
    return view_.at(indices...);
  }

  template <class... Indices>
  const T& at(Indices... indices) const {
    return view_.at(indices...);
  }

  template <class... Indices>
  size_t& get_1d_index(Indices... indices) {
    return view_.get_1d_index(indices...);
  }

  static size_t calculate_size(const std::array<std::size_t, sizeof...(lengths)>& dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
  }

  T* data() { return data_.data(); }

  const T* data() const { return data_.data(); }

  // 对齐后元素数量 为了simd加速
  size_t used_size() const { return data_.size(); }

  // 使用的元素数量
  size_t size() const { return raw_total_size; }

  std::array<size_t, sizeof...(lengths)> shapes() const { return view_.extents(); }

  std::array<size_t, sizeof...(lengths)> extents() const { return view_.extents(); }

  void set_value(T val) { std::fill(data_.begin(), data_.end(), val); }

  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() noexcept { return data_.begin(); }
  iterator end() noexcept { return data_.begin() + raw_total_size; }
  const_iterator begin() const noexcept { return data_.begin(); }
  const_iterator end() const noexcept { return data_.begin() + raw_total_size; }
  const_iterator cbegin() const noexcept { return data_.begin(); }
  const_iterator cend() const noexcept { return data_.begin() + raw_total_size; }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }
};

}  // namespace md

#endif  // __MDVECTOR_MDENGINE_STATIC_H__