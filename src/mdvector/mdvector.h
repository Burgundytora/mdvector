#ifndef HEADER_MDVECTOR_HPP_
#define HEADER_MDVECTOR_HPP_

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "allocator/aligned_allocator.h"
#include "exper_template/operator.h"
#include "simd/simd_function.h"
#include "span/mdspan.h"
#include "span/subspan.h"

// TODO: 行优先/列优先
struct layout_right {};
struct layout_left {};

// 核心mdvector类
template <class T, size_t Dims>
class mdvector : public Expr<mdvector<T, Dims>> {
 public:
  // ========================================================
  // 类成员
  std::array<size_t, Dims> dimensions_;       // 维度信息
  std::array<size_t, Dims> strides_;          // 索引偏置量
  size_t total_elements_ = 0;                 // 元素总数
  std::vector<T, AlignedAllocator<T>> data_;  // 数据

 public:
  // ========================================================
  // 构造函数  使用array静态维度数量
  mdvector(std::array<size_t, Dims> dim_set) : dimensions_{dim_set} {
    // 检查类型
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
    calculate_strides();
    data_.resize(total_elements_);
  }

  // 析构函数
  ~mdvector() = default;

  // ========================================================
  // 计算偏置 行优先
  void calculate_strides() {
    total_elements_ = std::reduce(dimensions_.begin(), dimensions_.end(), 1.0, std::multiplies<>());
    size_t stride = 1;
    for (size_t i = Dims - 1; i < Dims; --i) {  // 行优先倒序计算
      strides_[i] = stride;
      stride *= dimensions_[i];
    }
  }

  // ========================================================
  // 访问运算符 提供safe 和 unsafe两种方式
  // (i, j, k) unsafe style
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "mdvector dimension subscript wrong");
    const std::array<size_t, sizeof...(Indices)> idxs = {static_cast<size_t>(indices)...};
    if constexpr (Dims == 1) {
      return data_[idxs[0]];
    } else if constexpr (Dims == 2) {
      return data_[idxs[0] * strides_[0] + idxs[1]];
    } else {
      size_t offset = 0;
      for (size_t i = 0; i < Dims; i++) {
        offset += idxs[i] * strides_[i];
      }
      return data_[offset];
    }
  }

  // [i, j, k] unsafe style
  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "mdvector dimension subscript wrong");
    const std::array<size_t, sizeof...(Indices)> idxs = {static_cast<size_t>(indices)...};
    if constexpr (Dims == 1) {
      return data_[idxs[0]];
    } else if constexpr (Dims == 2) {
      return data_[idxs[0] * strides_[0] + idxs[1]];
    } else {
      size_t offset = 0;
      for (size_t i = 0; i < Dims; i++) {
        offset += idxs[i] * strides_[i];
      }
      return data_[offset];
    }
  }

  // .at(i, j, k) safe style
  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "mdvector dimension subscript wrong");
    size_t i = 0;
    for (auto len : std::array<size_t, Dims>{static_cast<size_t>(indices)...}) {
      if (len > dimensions_[i]) {
        std::cerr << "mdspan out-of-range error: " << len << ">" << dimensions_[i] << "\n";
        std::abort();
      }
      i++;
    }
    const std::array<size_t, sizeof...(Indices)> idxs = {static_cast<size_t>(indices)...};
    if constexpr (Dims == 1) {
      return data_[idxs[0]];
    } else if constexpr (Dims == 2) {
      return data_[idxs[0] * strides_[0] + idxs[1]];
    } else {
      size_t offset = 0;
      for (size_t i = 0; i < Dims; i++) {
        offset += idxs[i] * strides_[i];
      }
      return data_[offset];
    }
  }

  // ========================================================
  // 基础功能函数
  T* data() const { return const_cast<T*>(data_.data()); }

  size_t size() const { return total_elements_; }

  void set_value(T val) { std::fill(data_.begin(), data_.end(), val); }

  void show_data_array_style() {
    for (const auto& it : this->data_) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void show_data_matrix_style() {
    if (Dims == 0) return;

    const size_t cols = dimensions_.back();
    const size_t rows = data_.size() / cols;

    // std::cout << "data in matrix style:\n";
    for (size_t i = 0; i < rows; ++i) {
      const T* row_start = data_.data() + i * cols;
      for (size_t j = 0; j < cols; ++j) {
        std::cout << row_start[j] << " ";
      }
      std::cout << "\n";
    }
  }
  // ========================================================

  // TODO: 需要检查
  // 修改后的拷贝构造函数
  mdvector(const mdvector& other)
      : dimensions_(other.dimensions_),
        total_elements_(other.total_elements_),
        data_(other.data_),  // 直接复制数据
        strides_(other.strides_) {}

  // 添加深拷贝赋值运算符
  mdvector& operator=(const mdvector& other) {
    if (this != &other) {
      dimensions_ = other.dimensions_;
      total_elements_ = other.total_elements_;
      data_ = other.data_;  // 复制数据
      strides_ = other.strides_;
    }
    return *this;
  }

  // 添加移动构造函数
  mdvector(mdvector&& other) noexcept
      : dimensions_(std::move(other.dimensions_)),
        total_elements_(other.total_elements_),
        data_(std::move(other.data_)),
        strides_(std::move(data_.strides_)) {}

  // 添加移动赋值运算符
  mdvector& operator=(mdvector&& other) noexcept {
    if (this != &other) {
      dimensions_ = std::move(other.dimensions_);
      total_elements_ = other.total_elements_;
      data_ = std::move(other.data_);
      strides_ = std::move(other.strides_);
    }
    return *this;
  }

  // ========================================================
  // 用于表达式模板
  // 实现表达式赋值
  template <typename E>
  mdvector& operator=(const Expr<E>& expr) {
    expr.eval_to(this->data());  // 直接计算到目标内存
    return *this;
  }

  // 取值
  template <typename T2>
  typename simd<T2>::type eval_simd(size_t i) const {
    return simd<T2>::load(data_.data() + i);
  }

  // 取值
  template <typename T2>
  typename simd<T2>::type eval_simd_mask(size_t i) const {
    return simd<T2>::mask_load(data_.data() + i, total_elements_ - i);
  }

  // ========================================================
  // b ?= a
  mdvector& operator+=(const mdvector& other) {
    simd_add_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  mdvector& operator-=(const mdvector& other) {
    simd_sub_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  mdvector& operator*=(const mdvector& other) {
    simd_mul_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  mdvector& operator/=(const mdvector& other) {
    simd_div_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }
  // ========================================================

 private:
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
