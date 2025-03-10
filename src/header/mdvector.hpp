#ifndef HEADER_MDVECTOR_HPP_
#define HEADER_MDVECTOR_HPP_

#include <mdspan>
#include <array>
#include <vector>

#include "allocator.h"

// // 表达式模板基类 eigen设置思想
// // 为了复杂表达式引入性能下降，需要评估
// template <typename Derived>
// class Expr {
//  public:
//   auto operator[](size_t i) const { return static_cast<const Derived&>(*this)[i]; }
//   size_t size() const { return static_cast<const Derived&>(*this).size(); }
// };

// 核心MDVector类
template <typename T, size_t Dims>
class MDVector {
 public:
  // ========================================================
  // mdspan类型别名定义
  using extents_type = std::dextents<size_t, Dims>;
  using layout_type = std::layout_right;
  using mdspan_type = std::mdspan<T, extents_type, layout_type>;

 private:
  // ========================================================
  // AlignedAllocator<T> allocator_;
  // T* data_;
  std::vector<T, AlignedAllocator<T>> data_;
  mdspan_type view_;
  std::array<size_t, Dims> dimensions_;
  size_t total_elements_ = 0;

 public:
  // ========================================================
  // 构造函数
  template <typename... len>
  MDVector(len... dims) : dimensions_{static_cast<size_t>(dims)...} {
    static_assert(sizeof...(dims) == Dims, "mdvector dimension wrong");
    total_elements_ = 1;
    for (auto d : dimensions_) total_elements_ *= d;
    data_.resize(total_elements_);
    view_ = mdspan_type(data_, dimensions_);
  }

  // ========================================================
  // 访问运算符 提供safe 和 unsafe两种方式
  // (i, j, k) unsafe style
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "mdvector dimension subscript wrong");
    return view_[static_cast<size_t>(indices)...];
  }

  // [i, j, k] unsafe style
  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "mdvector dimension subscript wrong");
    return view_[static_cast<size_t>(indices)...];
  }

  // .at(i, j, k) safe style
  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "mdvector dimension subscript wrong");
    size_t i = 0;
    for (auto len : std::array<size_t, Dims>{static_cast<size_t>(indices)...}) {
      if (len > view_.extent(i)) {
        std::cerr << "mdspan out-of-range error: " << len << ">" << view_.extent(i) << "\n";
        std::abort();
      }
      i++;
    }
    return view_[indices...];
  }
  // ========================================================

  // 基础功能函数
  T* data() const { return data_; }

  size_t size() const { return total_elements_; }

  void SetValue(T val) { std::fill(data_.begin(), data_.end(), val); }

  void ShowDataArrayStyle() {
    // std::cout << "data in array style:\n";
    for (const auto& it : this->data_) {
      std::cout << it << " ";
    }
    std::cout << "\n";
  }

  void ShowDataMatrixStyle() {
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

  // 拷贝/赋值
  // data and view re build  data_ val equal
  MDVector& operator=(const MDVector& other) {
    *this = MDVector(other.dimensions_);
    this->data_ = other.data;
  }
  // data and view re build  data_ val not equal only dim equal
  MDVector(const MDVector& other) { *this = MDVector(other.dimensions_); }
  // ========================================================

  // 运算 avx2指令集方法
  MDVector operator+(const MDVector& other) {
    MDVector res(other);
    avx2_add(this->data_.data(), other.data_.data(), res.data_.data(), this->total_elements_);
    return res;
  }

  MDVector operator-(const MDVector& other) {
    MDVector res(other);
    avx2_sub(this->data_.data(), other.data_.data(), res.data_.data(), this->total_elements_);
    return res;
  }

  MDVector operator*(const MDVector& other) {
    MDVector res(other);
    avx2_mul(this->data_.data(), other.data_.data(), res.data_.data(), this->total_elements_);
    return res;
  }

  MDVector operator/(const MDVector& other) {
    MDVector res(other);
    avx2_div(this->data_.data(), other.data_.data(), res.data_.data(), this->total_elements_);
    return res;
  }

  // ========================================================
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
