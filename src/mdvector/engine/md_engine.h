#ifndef __MDVECTOR_MDEngine_H__
#define __MDVECTOR_MDEngine_H__

#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "../allocator/allocator.h"
#include "../exper_template/operator.h"
#include "../simd/simd_function.h"
#include "../span/mdspan.h"
#include "../span/subspan.h"

// 多维方法实现封装
template <class T, size_t Rank>
class MDEngine {
 protected:
  std::vector<T, AutoAllocator<T>> data_;
  mdspan<T, Rank> view_;

 public:
  // 默认构造
  MDEngine() = default;

  // 从维度array构造
  explicit MDEngine(const std::array<std::size_t, Rank>& dims)
      : data_(calculate_size(dims)), view_(mdspan<T, Rank>(data_.data(), dims)) {
    // 检查pod
    static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "T must be trivial and standard-layout!");
  }

  // 析构函数 成员全部为STL 默认析构即可
  ~MDEngine() = default;

  // 拷贝构造函数
  MDEngine(const MDEngine& other) : data_(other.data_), view_(data_.data(), other.view_.extents()) {}  // 初始化列表

  // 移动构造函数
  MDEngine(MDEngine&& other) noexcept : data_(std::move(other.data_)), view_(data_.data(), other.view_.extents()) {}

  // 深拷贝赋值运算符
  MDEngine& operator=(const MDEngine& other) {
    if (this != &other) {
      data_ = other.data_;                                           // 复制数据
      view_ = mdspan<T, Rank>(data_.data(), other.view_.extents());  // 视图重新创建
    }
    return *this;
  }

  // 移动赋值运算符
  MDEngine& operator=(MDEngine&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      view_ = mdspan<T, Rank>(data_.data(), other.view_.extents());
    }
    return *this;
  }

  // ========================================================
  // 多维访问
  template <class... Indices>
  T& operator()(Indices... indices) {
    return view_(indices...);
  }

  template <class... Indices>
  const T& operator()(Indices... indices) const {
    return view_(indices...);
  }

  // 安全访问
  template <class... Indices>
  T& at(Indices... indices) {
    return view_.at(indices...);
  }

  template <class... Indices>
  const T& at(Indices... indices) const {
    return view_.at(indices...);  // 假设 mdspan 有 at() 方法
  }

  // 计算多维索引的index偏移量
  template <class... Indices>
  size_t& get_1d_index(Indices... indices) {
    return view_.get_1d_index(indices...);
  }

  // 计算总元素数量
  static size_t calculate_size(const std::array<std::size_t, Rank>& dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
  }

  // ========================================================
  // 基础信息访问功能
  T* data() { return data_.data(); }

  const T* data() const { return data_.data(); }

  size_t size() const { return data_.size(); }

  std::array<size_t, Rank> shapes() const { return view_.extents(); }

  std::array<size_t, Rank> extents() const { return view_.extents(); }
  // ========================================================

  // ====================== 迭代器 ============================
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  // ====================== 正向迭代器 ======================
  iterator begin() noexcept { return data_.data(); }
  iterator end() noexcept { return data_.data() + data_.size(); }

  const_iterator begin() const noexcept { return data_.data(); }
  const_iterator end() const noexcept { return data_.data() + data_.size(); }

  // ====================== 常量迭代器 (C++11风格) ======================
  const_iterator cbegin() const noexcept { return data_.data(); }
  const_iterator cend() const noexcept { return data_.data() + data_.size(); }

  // ====================== 反向迭代器 ======================
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

  // ====================== 常量反向迭代器 ======================
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

  // ======================= 基础功能函数 ======================
  // 填充
  void set_value(T val) { std::fill(data_.begin(), data_.end(), val); }

  // 重置维度
  void reset_shape(const std::array<std::size_t, Rank>& dims) {
    data_.resize(calculate_size(dims));
    view_ = mdspan<T, Rank>(data_.data(), dims);
  }

  // ======================= 切片操作 ============================
  template <class... Slices>
  subspan<T, Rank> create_subspan(Slices... slices) {
    static_assert(sizeof...(Slices) == Rank, "Number of slices must match dimensionality");

    // 确保所有切片都转换为 md::slice 类型
    auto slice_array = prepare_slices(slices...);

    // 创建子视图
    return subspan<T, Rank>(data_.data(), view_.extents(), slice_array);
  }

  template <class... Slices>
  std::array<md::slice, Rank> prepare_slices(Slices... slices) {
    std::array<md::slice, Rank> result;
    size_t i = 0;

    // 使用折叠表达式处理每个切片
    ((result[i++] = convert_slice(slices)), ...);

    return result;
  }

  template <class SliceType>
  md::slice convert_slice(SliceType&& slice_one) {
    if constexpr (std::is_same_v<std::decay_t<SliceType>, md::slice>) {
      return std::forward<SliceType>(slice_one);
    } else if constexpr (std::is_integral_v<std::decay_t<SliceType>>) {
      // 整数索引转换为单元素切片
      return md::slice(static_cast<std::ptrdiff_t>(slice_one), static_cast<std::ptrdiff_t>(slice_one), false);
    } else {
      static_assert(sizeof(SliceType) == 0, "Unsupported slice type");
    }
  }
};

#endif  // __MDEngine_H__