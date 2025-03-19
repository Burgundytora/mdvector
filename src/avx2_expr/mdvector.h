#ifndef HEADER_MDVector22_HPP_
#define HEADER_MDVector22_HPP_

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "allocator.h"
#include "function.h"

// 维度数量设置
using MDShape_1d = std::array<size_t, 1>;
using MDShape_2d = std::array<size_t, 2>;
using MDShape_3d = std::array<size_t, 3>;
using MDShape_4d = std::array<size_t, 4>;

// // 表达式模板基类 eigen设置思想
// // 抽象类封装后 相比手写avx2性能下降10%~20% 小数据损失更多

#include "mask.h"
#include "simd_config.h"

// ======================== 表达式模板基类 ========================
// 表达式模板基类（抽象接口）
template <typename E>
class Expr2 {
 public:
  // 表达式求值：将结果写入目标内存
  void eval_to(double* dst) const { static_cast<const E&>(*this).eval(dst); }

  // 获取表达式大小
  size_t size() const { return static_cast<const E&>(*this).size(); }
};

// ======================== 表达式类 ========================
template <typename E1, typename E2>
class VectorAdd : public Expr2<VectorAdd<E1, E2>> {
 public:
  VectorAdd(const E1& e1, const E2& e2) : e1_(e1), e2_(e2) {}

  // 求值：遍历元素并相加
  void eval(double* dst) const {
    for (size_t i = 0; i <= size() - 4; i += 4) {  // 一次处理 4 个元素（AVX 双精度）
      __m256d a = _mm256_load_pd(e1_.data() + i);
      __m256d b = _mm256_load_pd(e2_.data() + i);
      __m256d c = _mm256_add_pd(a, b);
      _mm256_store_pd(dst + i, c);
    }
  }

  size_t size() const { return e1_.size(); }

 private:
  const E1& e1_;
  const E2& e2_;
};

template <typename E1, typename E2>
class VectorSub : public Expr2<VectorSub<E1, E2>> {
 public:
  VectorSub(const E1& e1, const E2& e2) : e1_(e1), e2_(e2) {}

  // 求值：遍历元素并相加
  void eval(double* dst) const {
    for (size_t i = 0; i <= size() - 4; i += 4) {  // 一次处理 4 个元素（AVX 双精度）
      __m256d a = _mm256_load_pd(e1_.data() + i);
      __m256d b = _mm256_load_pd(e2_.data() + i);
      __m256d c = _mm256_sub_pd(a, b);
      _mm256_store_pd(dst + i, c);
    }
  }

  size_t size() const { return e1_.size(); }

 private:
  const E1& e1_;
  const E2& e2_;
};

template <typename E1, typename E2>
class VectorMul : public Expr2<VectorMul<E1, E2>> {
 public:
  VectorMul(const E1& e1, const E2& e2) : e1_(e1), e2_(e2) {}

  // 求值：遍历元素并相加
  void eval(double* dst) const {
    for (size_t i = 0; i <= size() - 4; i += 4) {  // 一次处理 4 个元素（AVX 双精度）
      __m256d a = _mm256_load_pd(e1_.data() + i);
      __m256d b = _mm256_load_pd(e2_.data() + i);
      __m256d c = _mm256_mul_pd(a, b);
      _mm256_store_pd(dst + i, c);
    }
  }

  size_t size() const { return e1_.size(); }

 private:
  const E1& e1_;
  const E2& e2_;
};

template <typename E1, typename E2>
class VectorDiv : public Expr2<VectorDiv<E1, E2>> {
 public:
  VectorDiv(const E1& e1, const E2& e2) : e1_(e1), e2_(e2) {}

  // 求值：遍历元素并相加
  void eval(double* dst) const {
    for (size_t i = 0; i <= size() - 4; i += 4) {  // 一次处理 4 个元素（AVX 双精度）
      __m256d a = _mm256_load_pd(e1_.data() + i);
      __m256d b = _mm256_load_pd(e2_.data() + i);
      __m256d c = _mm256_div_pd(a, b);
      _mm256_store_pd(dst + i, c);
    }
  }

  size_t size() const { return e1_.size(); }

 private:
  const E1& e1_;
  const E2& e2_;
};

// 核心MDVector2类
template <class T, size_t Dims>
class MDVector2 : public Expr2<MDVector2<T, Dims>> {
 public:
  // ========================================================
  // 类成员
  std::array<size_t, Dims> dimensions_;        // 维度信息
  std::array<size_t, Dims> strides_;           // 索引偏置量
  size_t total_elements_ = 0;                  // 元素总数
  std::vector<T, AlignedAllocator2<T>> data_;  // 数据

 public:
  // ========================================================
  // 构造函数  使用array静态维度数量
  MDVector2(std::array<size_t, Dims> dim_set) : dimensions_{dim_set} {
    // 检查类型
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
    calculate_strides();
    data_.resize(total_elements_);
  }

  // 析构函数
  ~MDVector2() = default;

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
    static_assert(sizeof...(Indices) == Dims, "MDVector2 dimension subscript wrong");
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
    static_assert(sizeof...(Indices) == Dims, "MDVector2 dimension subscript wrong");
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
    static_assert(sizeof...(Indices) == Dims, "MDVector2 dimension subscript wrong");
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
  MDVector2(const MDVector2& other)
      : dimensions_(other.dimensions_),
        total_elements_(other.total_elements_),
        data_(other.data_),  // 直接复制数据
        strides_(other.strides_) {}

  // // 添加深拷贝赋值运算符
  // MDVector2& operator=(const MDVector2& other) {
  //   if (this != &other) {
  //     dimensions_ = other.dimensions_;
  //     total_elements_ = other.total_elements_;
  //     data_ = other.data_;  // 复制数据
  //     strides_ = other.strides_;
  //   }
  //   return *this;
  // }

  // 添加移动构造函数
  MDVector2(MDVector2&& other) noexcept
      : dimensions_(std::move(other.dimensions_)),
        total_elements_(other.total_elements_),
        data_(std::move(other.data_)),
        strides_(std::move(data_.strides_)) {}

  // // 添加移动赋值运算符
  // MDVector2& operator=(MDVector2&& other) noexcept {
  //   if (this != &other) {
  //     dimensions_ = std::move(other.dimensions_);
  //     total_elements_ = other.total_elements_;
  //     data_ = std::move(other.data_);
  //     strides_ = std::move(other.strides_);
  //   }
  //   return *this;
  // }

  // // 向量赋值操作符重载：触发表达式求值
  // template <typename T, size_t Dims, typename E>
  // MDVector<T, Dims>& operator=(MDVector<T, Dims>& lhs, const Expr2<E>& Expr2) {
  //   Expr2.eval_to(lhs.data());
  //   return lhs;
  // }

  // // 实现表达式赋值
  // template <typename E>
  // FORCE_INLINE MDVector2& operator=(const Expr<E>& expr) {
  //   expr.eval_to(this->data());  // 直接计算到目标内存
  //   return *this;
  // }

  // // 取值
  // template <typename T2>
  // FORCE_INLINE typename SimdConfig<T2>::simd_type eval_simd(size_t i) const {
  //   if constexpr (std::is_same_v<T, float>) {
  //     return _mm256_load_ps(data_.data() + i);
  //   } else {
  //     return _mm256_load_pd(data_.data() + i);
  //   }
  // }

  // // 取值
  // template <typename T2>
  // FORCE_INLINE typename SimdConfig<T2>::simd_type eval_simd_mask(size_t i) const {
  //   if constexpr (std::is_same_v<T, float>) {
  //     return _mm256_maskload_ps(data_.data() + i, mask_table_8[total_elements_ - i]);
  //   } else {
  //     return _mm256_maskload_pd(data_.data() + i, mask_table_4[total_elements_ - i]);
  //   }
  // }

  // ========================================================
  // 函数形式 有时候比表达式模板快一些
  // c = a + b
  // c.equal_a_add_b(a, b)
  void equal_a_add_b(const MDVector2& a, const MDVector2& b) {
    avx2_add(a.data(), b.data(), this->data(), this->total_elements_);
  }
  void equal_a_add_b(const MDVector2& a, const T& b) {
    avx2_add_scalar(a.data(), b, this->data(), this->total_elements_);
  }
  void equal_a_add_b(const T& a, const MDVector2& b) {
    avx2_add_scalar(b.data(), a, this->data(), this->total_elements_);
  }

  // c = a - b
  // c.equal_a_sub_b(a, b)
  void equal_a_sub_b(const MDVector2& a, const MDVector2& b) {
    avx2_sub(a.data(), b.data(), this->data(), this->total_elements_);
  }
  void equal_a_sub_b(const MDVector2& a, const T& b) {
    avx2_sub_scalar(a.data(), b, this->data(), this->total_elements_);
  }
  void equal_a_sub_b(const T& a, const MDVector2& b) {
    avx2_sub_scalar(b.data(), a, this->data(), this->total_elements_);
  }

  // c = a * b
  // c.equal_a_mul_b(a, b)
  void equal_a_mul_b(const MDVector2& a, const MDVector2& b) {
    avx2_mul(a.data(), b.data(), this->data(), this->total_elements_);
  }
  void equal_a_mul_b(const MDVector2& a, const T& b) {
    avx2_mul_scalar(a.data(), b, this->data(), this->total_elements_);
  }
  void equal_a_mul_b(const T& a, const MDVector2& b) {
    avx2_mul_scalar(b.data(), a, this->data(), this->total_elements_);
  }

  // c = a / b
  // c.equal_a_div_b(a, b)
  void equal_a_div_b(const MDVector2& a, const MDVector2& b) {
    avx2_div(a.data(), b.data(), this->data(), this->total_elements_);
  }
  void equal_a_div_b(const MDVector2& a, const T& b) {
    avx2_div_scalar(a.data(), b, this->data(), this->total_elements_);
  }
  void equal_a_div_b(const T& a, const MDVector2& b) {
    avx2_mul_scalar(b.data(), 1.0 / a, this->data(), this->total_elements_);
  }
  // ========================================================

  // ========================================================
  // b ?= a
  MDVector2& operator+=(const MDVector2& other) {
    avx2_add_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  MDVector2& operator-=(const MDVector2& other) {
    avx2_sub_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  MDVector2& operator*=(const MDVector2& other) {
    avx2_mul_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  MDVector2& operator/=(const MDVector2& other) {
    avx2_div_inplace(other.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }
  // ========================================================

  // FMA加乘融合
  // ?=a*b+c
  // MDVector2 new = fma_abc(a, b, c)
  MDVector2 fma_abc(const MDVector2& a, const MDVector2& b, const MDVector2& c) {
    MDVector2 d(a);
    avx2_fma(a.data_.data(), b.data_.data(), c.data_.data(), d.data_.data(), this->total_elements_);
    return d;
  }

  // c=a*b+c
  // c.fma_c_abc(a, b)
  MDVector2& fma_c_abc(const MDVector2& a, const MDVector2& b) {
    avx2_fma(a.data_.data(), b.data_.data(), this->data_.data(), this->total_elements_);
    return *this;
  }

  // 关键修正：成员函数形式的 operator= 重载
  template <typename E>
  MDVector2<T, Dims>& operator=(const Expr2<E>& expr) {
    expr.eval_to(data_.data());  // 计算表达式并写入当前对象
    return *this;
  }

 private:
};

// ======================== 运算符重载 ========================
// 加法操作符重载：返回 VectorAdd 表达式
template <typename E1, typename E2>
VectorAdd<E1, E2> operator+(const Expr2<E1>& e1, const Expr2<E2>& e2) {
  return VectorAdd<E1, E2>(static_cast<const E1&>(e1), static_cast<const E2&>(e2));
}

template <typename E1, typename E2>
VectorSub<E1, E2> operator-(const Expr2<E1>& e1, const Expr2<E2>& e2) {
  return VectorSub<E1, E2>(static_cast<const E1&>(e1), static_cast<const E2&>(e2));
}

template <typename E1, typename E2>
VectorSub<E1, E2> operator*(const Expr2<E1>& e1, const Expr2<E2>& e2) {
  return VectorSub<E1, E2>(static_cast<const E1&>(e1), static_cast<const E2&>(e2));
}

template <typename E1, typename E2>
VectorDiv<E1, E2> operator/(const Expr2<E1>& e1, const Expr2<E2>& e2) {
  return VectorDiv<E1, E2>(static_cast<const E1&>(e1), static_cast<const E2&>(e2));
}

#endif  // HEADER_MDVector22_HPP_
