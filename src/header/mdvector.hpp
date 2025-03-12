#ifndef HEADER_MDVECTOR_HPP_
#define HEADER_MDVECTOR_HPP_

#include <mdspan>
#include <array>
#include <vector>

#include "avx2.h"

// 维度数量设置
using MDShape_1d = std::array<size_t, 1>;
using MDShape_2d = std::array<size_t, 2>;
using MDShape_3d = std::array<size_t, 3>;
using MDShape_4d = std::array<size_t, 4>;

// // 表达式模板基类 eigen设置思想
// // 抽象类封装后 相比手写avx2性能下降10%~20% 小数据损失更多

// 核心MDVector类
template <class T, size_t Dims>
class MDVector : public Expr<MDVector<T, Dims>> {
 public:
  // ========================================================
  // mdspan类型别名定义
  using extents_type = std::dextents<size_t, Dims>;
  using layout_type = std::layout_right;
  using mdspan_type = std::mdspan<T, extents_type, layout_type>;

 public:
  // ========================================================
  // 类成员
  std::vector<T, AlignedAllocator<T>> data_;  // 数据
  mdspan_type view_;                          // 一维vector的多维视图
  std::array<size_t, Dims> dimensions_;       // 维度信息
  size_t total_elements_ = 0;                 // 元素总数

  template <size_t... I>
  extents_type CreateExtents(std::index_sequence<I...>) {
    return extents_type{dimensions_[I]...};
  }

 public:
  // ========================================================
  // 构造函数  使用array静态维度数量
  MDVector(std::array<size_t, Dims> dim_set) : dimensions_{dim_set} {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
    total_elements_ = 1;
    for (auto d : dimensions_) {
      total_elements_ *= d;
    }
    data_.resize(total_elements_);
    view_ = mdspan_type(data_.data(), CreateExtents(std::make_index_sequence<Dims>{}));
  }
  // 析构函数
  ~MDVector() = default;

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
  T* data() const { return const_cast<T*>(data_.data()); }

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

  // TODO: 需要检查
  // 修改后的拷贝构造函数
  MDVector(const MDVector& other)
      : dimensions_(other.dimensions_),
        total_elements_(other.total_elements_),
        data_(other.data_),  // 直接复制数据
        view_(data_.data(), CreateExtents(std::make_index_sequence<Dims>{})) {}

  // 添加深拷贝赋值运算符
  MDVector& operator=(const MDVector& other) {
    if (this != &other) {
      dimensions_ = other.dimensions_;
      total_elements_ = other.total_elements_;
      data_ = other.data_;  // 复制数据
      view_ = mdspan_type(data_, CreateExtents(std::make_index_sequence<Dims>{}));
    }
    return *this;
  }

  // 添加移动构造函数
  MDVector(MDVector&& other) noexcept
      : dimensions_(std::move(other.dimensions_)),
        total_elements_(other.total_elements_),
        data_(std::move(other.data_)),
        view_(data_.data(), CreateExtents(std::make_index_sequence<Dims>{})) {}

  // 添加移动赋值运算符
  MDVector& operator=(MDVector&& other) noexcept {
    if (this != &other) {
      dimensions_ = std::move(other.dimensions_);
      total_elements_ = other.total_elements_;
      data_ = std::move(other.data_);
      data_ = other.data_;
      view_ = mdspan_type(data_.data(), CreateExtents(std::make_index_sequence<Dims>{}));
    }
    return *this;
  }

  // 实现表达式赋值
  template <typename E>
  MDVector& operator=(const Expr<E>& expr) {
    expr.eval_to(this->data());  // 直接计算到目标内存
    return *this;
  }

  // 实现表达式求值
  void eval_to_impl(T* __restrict dest) const { avx2_copy(this->data_, dest, this->size()); }
  // ========================================================

  // 函数形式 有时候比表达式模板快一些
  // c = a + b
  // c.equal_a_add_b(a, b)
  void equal_a_add_b(const MDVector& a, const MDVector& b) {
    avx2_add(a.data(), b.data(), this->data(), this->total_elements_);
  }

  // c = a - b
  // c.equal_a_sub_b(a, b)
  void equal_a_sub_b(const MDVector& a, const MDVector& b) {
    avx2_sub(a.data(), b.data(), this->data(), this->total_elements_);
  }

  // c = a * b
  // c.equal_a_mul_b(a, b)
  void equal_a_mul_b(const MDVector& a, const MDVector& b) {
    avx2_mul(a.data(), b.data(), this->data(), this->total_elements_);
  }

  // c = a / b
  // c.equal_a_div_b(a, b)
  void equal_a_div_b(const MDVector& a, const MDVector& b) {
    avx2_div(a.data(), b.data(), this->data(), this->total_elements_);
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
