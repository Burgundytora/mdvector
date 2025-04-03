#ifndef HEADER_MDVECTOR_HPP_
#define HEADER_MDVECTOR_HPP_

#include "engine/md_engine.h"

// =================================================
// 针对非浮点类型 具备多维索引功能 不具备表达式计算功能
template <class T, size_t Rank, class Enable = void>
class mdvector : private MDEngine<T, Rank> {
  using Impl = MDEngine<T, Rank>;

 private:
  // 使用基础构造函数
  using Impl::Impl;

  // 其他基础需要显式转发
  mdvector(const mdvector& other) : Impl(other) {}

  mdvector(mdvector&& other) noexcept : Impl(std::move(other)) {}

  mdvector& operator=(const mdvector& other) {
    Impl::operator=(other);
    return *this;
  }

  mdvector& operator=(mdvector&& other) noexcept {
    Impl::operator=(std::move(other));
    return *this;
  }

  ~mdvector() = default;

  // ===================== 多维访问 ===========================
  using Impl::operator();
  using Impl::at;

  // =================== 基础信息访问功能 ======================
  using Impl::extents;
  using Impl::shapes;
  using Impl::size;

  // ====================== 迭代器 ============================
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  // 引入迭代器方法
  using Impl::begin;
  using Impl::cbegin;
  using Impl::cend;
  using Impl::crbegin;
  using Impl::crend;
  using Impl::end;
  using Impl::rbegin;
  using Impl::rend;

  // ======================= 基础功能函数 ======================
  using Impl::create_subspan;
  using Impl::reset_shape;
  using Impl::set_value;
};

// ================  浮点类型 支持元素级计算 (simd + ET)  ====================
//
template <class T, size_t Rank>
class mdvector<T, Rank, std::enable_if_t<std::is_floating_point_v<T>>> : public TensorExpr<mdvector<T, Rank>>,
                                                                         private MDEngine<T, Rank> {
  using Impl = MDEngine<T, Rank>;

 public:
  // 默认构造
  using Impl::Impl;

  // 其他基础需要显式转发
  mdvector(const mdvector& other) : Impl(other) {}

  mdvector(mdvector&& other) noexcept : Impl(std::move(other)) {}

  mdvector& operator=(const mdvector& other) {
    Impl::operator=(other);
    return *this;
  }

  mdvector& operator=(mdvector&& other) noexcept {
    Impl::operator=(std::move(other));
    return *this;
  }

  ~mdvector() = default;

  // ===================== 多维访问 ===========================
  using Impl::operator();
  using Impl::at;

  // =================== 基础信息访问功能 ======================
  using Impl::extents;
  using Impl::shapes;
  using Impl::size;

  // ====================== 迭代器 ============================
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  // 引入迭代器方法
  using Impl::begin;
  using Impl::cbegin;
  using Impl::cend;
  using Impl::crbegin;
  using Impl::crend;
  using Impl::end;
  using Impl::rbegin;
  using Impl::rend;

  // ======================= 基础功能函数 ======================
  using Impl::create_subspan;
  using Impl::reset_shape;
  using Impl::set_value;

  // ======================= 浮点类型特有功能 ======================

  // =================== 表达式模板 ============================
  // 表达式构造
  template <class E>
  mdvector(const TensorExpr<E>& expr) {
    this->reset_shape(expr.extents());
    expr.eval_to(this->data());  // 直接计算到目标内存
  }

  // 表达式赋值
  template <class E>
  mdvector& operator=(const TensorExpr<E>& expr) {
    expr.eval_to(this->data());  // 直接计算到目标内存
    return *this;
  }

  // 取值
  template <class T2>
  typename simd<T2>::type eval_simd(size_t i) const {
    return simd<T2>::load(data() + i);
  }

  // 取值
  template <class T2>
  typename simd<T2>::type eval_simd_mask(size_t i) const {
    return simd<T2>::mask_load(data() + i, size() - i);
  }

  // ======================= ?= 操作符重载 ============================
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

  // ====================== 标量表达式模板 ===========================
  // 添加标量eval_scalar方法
  template <class T2>
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

  // 变量打印
  void show_data_array_style() {
    for (const auto& it : this->data_) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void show_data_matrix_style() {
    if (Rank == 0) return;

    const size_t cols = view_.extent(Rank - 1);
    const size_t rows = size() / cols;

    for (size_t i = 0; i < rows; ++i) {
      const T* row_start = data() + i * cols;
      for (size_t j = 0; j < cols; ++j) {
        std::cout << row_start[j] << " ";
      }
      std::cout << "\n";
    }
  }
};

// ======================= 常用维度别名 1D~6D ============================
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
