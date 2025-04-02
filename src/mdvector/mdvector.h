#ifndef HEADER_MDVECTOR_HPP_
#define HEADER_MDVECTOR_HPP_

#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "allocator/aligned_allocator.h"
#include "exper_template/operator.h"
#include "simd/simd_function.h"
#include "span/mdspan.h"
#include "span/subspan.h"

// 核心mdvector类
template <class T, size_t Dims>
class mdvector : public Expr<mdvector<T, Dims>> {
 private:
  // ========================================================
  // 类成员
  mdspan<T, Dims> view_;                      // 多维视图
  std::vector<T, AlignedAllocator<T>> data_;  // 数据

 public:
  // 默认构造
  mdvector() = default;

  // 从维度array构造
  explicit mdvector(const std::array<std::size_t, Dims>& dims) : data_(calculate_size(dims)) {
    view_ = mdspan<T, Dims>(data_.data(), dims);
  }

  // 重置维度
  void reset(const std::array<std::size_t, Dims>& dims) {
    data_.resize(calculate_size(dims));
    view_ = mdspan<T, Dims>(data_.data(), dims);
  }

  // 析构函数 成员全部为STL 默认析构即可
  ~mdvector() = default;

  // ========================================================
  // 多维访问
  template <typename... Indices>
  T& operator()(Indices... indices) {
    return view_(indices...);
  }

  // 安全访问
  template <typename... Indices>
  T& at(Indices... indices) {
    return view_.at(indices...);
  }

  // 计算多维索引的index偏移量
  template <typename... Indices>
  size_t& get_1d_index(Indices... indices) {
    return view_.get_1d_index(indices...);
  }

  // ========================================================

  template <typename... Slices>
  auto create_subspan(Slices... slices) {
    static_assert(sizeof...(Slices) == Dims, "Number of slices must match dimensionality");

    // 确保所有切片都转换为 md::Slice 类型
    auto slice_array = prepare_slices(slices...);
    // for (const auto& s : slice_array) {
    //   std::cout << "start:" << s.start << " end:" << s.end << " is_all:" << s.is_all << std::endl;
    // }

    // 创建子视图
    return subspan<T, Dims>(data_.data(), view_.extents(), slice_array);
  }

  T* data() const { return const_cast<T*>(data_.data()); }

  size_t size() const { return data_.size(); }

  std::array<size_t, Dims> shape() const { return view_.shape(); }

  // ========================================================

  // 拷贝构造函数
  mdvector(const mdvector& other)
      : data_(other.data_)  // 初始化列表
  {
    view_ = mdspan<T, Dims>(data(), other.view_.extents());  // 保证在data_构造后
  }

  // 移动构造函数
  mdvector(mdvector&& other) noexcept : data_(std::move(other.data_)), view_(other.view_) {}

  // 深拷贝赋值运算符
  mdvector& operator=(const mdvector& other) {
    if (this != &other) {
      data_ = other.data_;                                           // 复制数据
      view_ = mdspan<T, Dims>(data_.data(), other.view_.extents());  // 视图重新创建
    }
    return *this;
  }

  // 移动赋值运算符
  mdvector& operator=(mdvector&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      view_ = other.view_;
    }
    return *this;
  }

  // =================== 表达式模板 ============================

  // 表达式构造
  template <typename E>
  mdvector(const Expr<E>& expr) {
    this->reset(expr.shape());
    expr.eval_to(this->data());  // 直接计算到目标内存
  }

  // 表达式赋值
  template <typename E>
  mdvector& operator=(const Expr<E>& expr) {
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
  static std::size_t calculate_size(const std::array<std::size_t, Dims>& dims) {
    std::size_t size = 1;
    for (auto d : dims) {
      size *= d;
    }
    return size;
  }

  // ====================== 迭代器 ============================

  T* begin() { return data_; }
  T* end() { return data_ + size(); }  // 尾后指针

  // const 重载
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size(); }

 public:
  // ========================================================
  // 基础功能函数
  void set_value(T val) { std::fill(data_.begin(), data_.end(), val); }

  void show_data_array_style() {
    for (const auto& it : this->data_) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void show_data_matrix_style() {
    if (Dims == 0) return;

    const size_t cols = view_.extent(Dims - 1);
    const size_t rows = size() / cols;

    // std::cout << "data in matrix style:\n";
    for (size_t i = 0; i < rows; ++i) {
      const T* row_start = data() + i * cols;
      for (size_t j = 0; j < cols; ++j) {
        std::cout << row_start[j] << " ";
      }
      std::cout << "\n";
    }
  }

  // ========================================================
  // b ?= a
  mdvector& operator+=(const mdvector& other) {
    simd_add_inplace(this->data(), other.data(), this->size());
    return *this;
  }

  mdvector& operator-=(const mdvector& other) {
    simd_sub_inplace(this->data(), other.data(), this->size());
    return *this;
  }

  mdvector& operator*=(const mdvector& other) {
    simd_mul_inplace(this->data(), other.data(), this->size());
    return *this;
  }

  mdvector& operator/=(const mdvector& other) {
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
  mdvector& operator+=(T scalar) {
    simd_add_inplace_scalar(this->data(), scalar, this->size());
    return *this;
  }

  mdvector& operator-=(T scalar) {
    simd_sub_inplace_scalar(this->data(), scalar, this->size());
    return *this;
  }

  mdvector& operator*=(T scalar) {
    simd_mul_inplace_scalar(this->data(), scalar, this->size());
    return *this;
  }

  mdvector& operator/=(T scalar) {
    simd_div_inplace_scalar(this->data(), scalar, this->size());
    return *this;
  }

  // ========================================================
  // 切片操作

  template <typename... Slices>
  std::array<md::slice, Dims> prepare_slices(Slices... slices) {
    std::array<md::slice, Dims> result;
    size_t i = 0;

    // 使用折叠表达式处理每个切片
    ((result[i++] = convert_slice(slices)), ...);

    return result;
  }

  template <typename SliceType>
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

// 常用维度别名 1D~6D

// 常用维度shape
using mdshape_1d = std::array<size_t, 1>;
using mdshape_2d = std::array<size_t, 2>;
using mdshape_3d = std::array<size_t, 3>;
using mdshape_4d = std::array<size_t, 4>;
using mdshape_5d = std::array<size_t, 5>;
using mdshape_6d = std::array<size_t, 6>;

// 常用维度mdvector
template <class T>
using mdvector_1d = mdvector<T, 1>;
template <class T>
using mdvector_2d = mdvector<T, 2>;
template <class T>
using mdvector_3d = mdvector<T, 3>;
template <class T>
using mdvector_4d = mdvector<T, 4>;
template <class T>
using mdvector_5d = mdvector<T, 5>;
template <class T>
using mdvector_6d = mdvector<T, 6>;

#endif  // HEADER_MDVECTOR_HPP_
