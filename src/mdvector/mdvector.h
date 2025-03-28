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
 private:
  // ========================================================
  // 类成员
  std::array<size_t, Dims> dimensions_;       // 维度信息
  std::array<size_t, Dims> strides_;          // 索引偏置量
  size_t total_elements_ = 0;                 // 元素总数
  std::vector<T, AlignedAllocator<T>> data_;  // 数据

 public:
  // 默认构造
  mdvector() = default;

  // ========================================================
  // 构造函数  使用array静态维度数量
  explicit mdvector(std::array<size_t, Dims> dim_set)
      : dimensions_(dim_set),
        strides_(calculate_strides()),
        total_elements_(calculate_total_elements()),
        data_(calculate_total_elements())  // 初始化data_的大小
  {
    // 类型检查
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "mdvector Type must be float or double!");
  }

  void Reset(std::array<size_t, Dims> dim_set) {
    this->~mdvector();
    *this = mdvector(dim_set);
  }

  // 析构函数 成员全部为STL 默认析构即可
  ~mdvector() = default;

  // ========================================================

  T* data() const { return const_cast<T*>(data_.data()); }

  size_t size() const { return total_elements_; }

  std::array<size_t, Dims> shape() const { return dimensions_; }

  // ========================================================

  // TODO: 需要检查
  // 拷贝构造函数
  mdvector(const mdvector& other)
      : dimensions_(other.dimensions_),
        total_elements_(other.total_elements_),
        data_(other.data_),  // 直接复制数据
        strides_(other.strides_) {}

  // 移动构造函数
  mdvector(mdvector&& other) noexcept
      : dimensions_(std::move(other.dimensions_)),
        total_elements_(other.total_elements_),
        data_(std::move(other.data_)),
        strides_(std::move(data_.strides_)) {}

  // 深拷贝赋值运算符
  mdvector& operator=(const mdvector& other) {
    if (this != &other) {
      dimensions_ = other.dimensions_;
      total_elements_ = other.total_elements_;
      data_ = other.data_;  // 复制数据
      strides_ = other.strides_;
    }
    return *this;
  }

  // 移动赋值运算符
  mdvector& operator=(mdvector&& other) noexcept {
    if (this != &other) {
      dimensions_ = std::move(other.dimensions_);
      total_elements_ = other.total_elements_;
      data_ = std::move(other.data_);
      strides_ = std::move(other.strides_);
    }
    return *this;
  }

  // =================== 表达式模板 ============================

  // 表达式构造
  template <typename E>
  mdvector(const Expr<E>& expr) {
    this->Reset(expr.shape());
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
    return simd<T2>::mask_load(data() + i, total_elements_ - i);
  }

  // ========================================================
  // 计算总元素数和步长
  size_t calculate_total_elements() const {
    size_t total = 1;
    for (auto dim : dimensions_) {
      total *= dim;
    }
    return total;
  }

  // 计算偏置 行优先
  std::array<size_t, Dims> calculate_strides() const {
    std::array<size_t, Dims> strides;
    strides.back() = 1;  // 最后一个维度步长为1
    for (int i = Dims - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dimensions_[i + 1];
    }
    return strides;
  }

  // ========================================================
  // 访问运算符 提供safe 和 unsafe两种方式
  // (i, j, k) unsafe style
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "mdvector dimension subscript wrong");
    const std::array<size_t, sizeof...(Indices)> idxs = {static_cast<size_t>(indices)...};
    size_t offset = 0;
    for (size_t i = 0; i < Dims; i++) {
      offset += idxs[i] * strides_[i];
    }
    return data_[offset];
  }

  // .at(i, j, k) safe style
  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "mdvector dimension subscript wrong");
    const std::array<size_t, sizeof...(Indices)> idxs = {static_cast<size_t>(indices)...};
    size_t i_temp = 0;
    for (auto len : idxs) {
      if (len > dimensions_[i_temp]) {
        std::cerr << "mdvector subscript out-of-range error: " << len << ">" << dimensions_[i_temp] << "\n";
        std::abort();
      }
      i_temp++;
    }

    size_t offset = 0;
    for (size_t i = 0; i < Dims; i++) {
      offset += idxs[i] * strides_[i];
    }
    return data_[offset];
  }

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

    const size_t cols = dimensions_.back();
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
    simd_add_inplace(this->data(), other.data(), this->total_elements_);
    return *this;
  }

  mdvector& operator-=(const mdvector& other) {
    simd_sub_inplace(this->data(), other.data(), this->total_elements_);
    return *this;
  }

  mdvector& operator*=(const mdvector& other) {
    simd_mul_inplace(this->data(), other.data(), this->total_elements_);
    return *this;
  }

  mdvector& operator/=(const mdvector& other) {
    simd_div_inplace(this->data(), other.data(), this->total_elements_);
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
