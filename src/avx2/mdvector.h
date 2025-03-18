#ifndef HEADER_MDVECTOR_HPP_
#define HEADER_MDVECTOR_HPP_

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "allocator.h"
#include "function.h"
#include "operator.h"


// 维度数量设置
using MDShape_1d = std::array<size_t, 1>;
using MDShape_2d = std::array<size_t, 2>;
using MDShape_3d = std::array<size_t, 3>;
using MDShape_4d = std::array<size_t, 4>;

// 行优先/列优先
struct layout_right {};
struct layout_left {};

// // 表达式模板基类 eigen设置思想
// // 抽象类封装后 相比手写avx2性能下降10%~20% 小数据损失更多

// 核心MDVector类
template <class T, size_t Dims, class LayoutPolicy = layout_right>
class MDVector : public Expr<MDVector<T, Dims>> {
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
  MDVector(std::array<size_t, Dims> dim_set) : dimensions_{dim_set} {
    // 检查类型
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
    calculate_strides();
    data_.resize(total_elements_);
  }

  // 析构函数
  ~MDVector() = default;

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

    } else if constexpr (Dims == 3) {
      return data_[idxs[0] * strides_[0] + idxs[1] * strides_[1] + idxs[2]];

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

    } else if constexpr (Dims == 3) {
      return data_[idxs[0] * strides_[0] + idxs[1] * strides_[1] + idxs[2]];

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

    } else if constexpr (Dims == 3) {
      return data_[idxs[0] * strides_[0] + idxs[1] * strides_[1] + idxs[2]];

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
  MDVector(const MDVector& other)
      : dimensions_(other.dimensions_),
        total_elements_(other.total_elements_),
        data_(other.data_),  // 直接复制数据
        strides_(other.strides_) {}

  // 添加深拷贝赋值运算符
  MDVector& operator=(const MDVector& other) {
    if (this != &other) {
      dimensions_ = other.dimensions_;
      total_elements_ = other.total_elements_;
      data_ = other.data_;  // 复制数据
      strides_ = other.strides_;
    }
    return *this;
  }

  // 添加移动构造函数
  MDVector(MDVector&& other) noexcept
      : dimensions_(std::move(other.dimensions_)),
        total_elements_(other.total_elements_),
        data_(std::move(other.data_)),
        strides_(std::move(data_.strides_)) {}

  // 添加移动赋值运算符
  MDVector& operator=(MDVector&& other) noexcept {
    if (this != &other) {
      dimensions_ = std::move(other.dimensions_);
      total_elements_ = other.total_elements_;
      data_ = std::move(other.data_);
      strides_ = std::move(other.strides_);
    }
    return *this;
  }

  // 实现表达式赋值
  template <typename E>
  FORCE_INLINE MDVector& operator=(const Expr<E>& expr) {
    expr.eval_to(this->data());  // 直接计算到目标内存
    return *this;
  }

  // 取值
  template <typename T2>
  FORCE_INLINE typename SimdConfig<T2>::simd_type eval_simd(size_t i) const {
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_load_ps(data_.data() + i);
    } else {
      return _mm256_load_pd(data_.data() + i);
    }
  }

  // 取值
  template <typename T2>
  FORCE_INLINE typename SimdConfig<T2>::simd_type eval_simd_mask(size_t i) const {
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_maskload_ps(data_.data() + i, mask_table_8[total_elements_ - i]);
    } else {
      return _mm256_maskload_pd(data_.data() + i, mask_table_4[total_elements_ - i]);
    }
  }

  // ========================================================
  // 函数形式 有时候比表达式模板快一些
  // c = a + b
  // c.equal_a_add_b(a, b)
  void equal_a_add_b(const MDVector& a, const MDVector& b) {
    avx2_add(a.data(), b.data(), this->data(), this->total_elements_);
  }
  void equal_a_add_b(const MDVector& a, const T& b) {
    avx2_add_scalar(a.data(), b, this->data(), this->total_elements_);
  }
  void equal_a_add_b(const T& a, const MDVector& b) {
    avx2_add_scalar(b.data(), a, this->data(), this->total_elements_);
  }

  // c = a - b
  // c.equal_a_sub_b(a, b)
  void equal_a_sub_b(const MDVector& a, const MDVector& b) {
    avx2_sub(a.data(), b.data(), this->data(), this->total_elements_);
  }
  void equal_a_sub_b(const MDVector& a, const T& b) {
    avx2_sub_scalar(a.data(), b, this->data(), this->total_elements_);
  }
  void equal_a_sub_b(const T& a, const MDVector& b) {
    avx2_sub_scalar(b.data(), a, this->data(), this->total_elements_);
  }

  // c = a * b
  // c.equal_a_mul_b(a, b)
  void equal_a_mul_b(const MDVector& a, const MDVector& b) {
    avx2_mul(a.data(), b.data(), this->data(), this->total_elements_);
  }
  void equal_a_mul_b(const MDVector& a, const T& b) {
    avx2_mul_scalar(a.data(), b, this->data(), this->total_elements_);
  }
  void equal_a_mul_b(const T& a, const MDVector& b) {
    avx2_mul_scalar(b.data(), a, this->data(), this->total_elements_);
  }

  // c = a / b
  // c.equal_a_div_b(a, b)
  void equal_a_div_b(const MDVector& a, const MDVector& b) {
    avx2_div(a.data(), b.data(), this->data(), this->total_elements_);
  }
  void equal_a_div_b(const MDVector& a, const T& b) {
    avx2_div_scalar(a.data(), b, this->data(), this->total_elements_);
  }
  void equal_a_div_b(const T& a, const MDVector& b) {
    avx2_mul_scalar(b.data(), 1.0 / a, this->data(), this->total_elements_);
  }
  // ========================================================

  // ========================================================
  // b ?= a
  MDVector& operator+=(const MDVector& other) {
    avx2_add_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  MDVector& operator-=(const MDVector& other) {
    avx2_sub_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  MDVector& operator*=(const MDVector& other) {
    avx2_mul_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  MDVector& operator/=(const MDVector& other) {
    avx2_div_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }
  // ========================================================

  // FMA加乘融合
  // ?=a*b+c
  // MDVector new = fma_abc(a, b, c)
  MDVector fma_abc(const MDVector& a, const MDVector& b, const MDVector& c) {
    MDVector d(a);
    avx2_fma(a.data_.data(), b.data_.data(), c.data_.data(), d.data_.data(), this->total_elements_);
    return d;
  }

  // c=a*b+c
  // c.fma_c_abc(a, b)
  MDVector& fma_c_abc(const MDVector& a, const MDVector& b) {
    avx2_fma(a.data_.data(), b.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

 private:
};

#endif  // HEADER_MDVECTOR_HPP_
