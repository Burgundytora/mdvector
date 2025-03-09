#include <array>
#include <immintrin.h>
#include <mkl.h>
#include <memory>
#include <type_traits>
#include <utility>

// 编译开关定义
// #define MDVECTOR_USE_MKL
// #define MDVECTOR_USE_AVX2

// 内存对齐分配器
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  static constexpr size_t alignment =
#ifdef MDVECTOR_USE_MKL
      64;  // MKL推荐对齐
#else
      32;  // AVX2基础对齐
#endif

  T* allocate(size_t n) {
    size_t aligned_size = align_size(n);
    void* ptr =
#ifdef MDVECTOR_USE_MKL
        mkl_malloc(aligned_size * sizeof(T), alignment);
#else
        aligned_alloc(alignment, aligned_size * sizeof(T));
#endif
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, size_t) {
#ifdef MDVECTOR_USE_MKL
    mkl_free(p);
#else
    free(p);
#endif
  }

 private:
  static constexpr size_t pack_size() {
    if constexpr (std::is_same_v<T, float>) {
#ifdef MDVECTOR_USE_AVX2
      return 8;  // AVX2单精度打包数
#else
      return 1;
#endif
    } else {
#ifdef MDVECTOR_USE_AVX2
      return 4;  // AVX2双精度打包数
#else
      return 1;
#endif
    }
  }

  static size_t align_size(size_t n) { return ((n + pack_size() - 1) / pack_size()) * pack_size(); }
};

// 表达式模板基类
template <typename Derived>
class Expr {
 public:
  auto operator[](size_t i) const { return static_cast<const Derived&>(*this)[i]; }
  size_t size() const { return static_cast<const Derived&>(*this).size(); }
};

// 具体张量类
template <typename T, size_t N>
class MDVector : public Expr<MDVector<T, N>> {
 public:
  // 构造函数
  template <typename... Dims>
  MDVector(Dims... dims) : dimensions_{static_cast<size_t>(dims)...} {
    static_assert(sizeof...(Dims) == N, "维度数量不匹配");
    total_elements_ = 1;
    for (auto d : dimensions_) total_elements_ *= d;
    aligned_size_ = allocator_.align_size(total_elements_);
    data_ = allocator_.allocate(aligned_size_);
  }

  // 操作符重载
  template <typename E>
  MDVector& operator+=(const Expr<E>& expr) {
    for (size_t i = 0; i < aligned_size_; ++i) {
      data_[i] += expr[i];
    }
    return *this;
  }

  template <typename E>
  MDVector& operator-=(const Expr<E>& expr) {
    for (size_t i = 0; i < aligned_size_; ++i) {
      data_[i] -= expr[i];
    }
    return *this;
  }

  template <typename E>
  MDVector& operator*=(const Expr<E>& expr) {
    for (size_t i = 0; i < aligned_size_; ++i) {
      data_[i] *= expr[i];
    }
    return *this;
  }

  template <typename E>
  MDVector& operator/=(const Expr<E>& expr) {
    for (size_t i = 0; i < aligned_size_; ++i) {
      data_[i] /= expr[i];
    }
    return *this;
  }

  // 表达式模板支持
  template <typename E1, typename E2, typename Op>
  class BinaryExpr : public Expr<BinaryExpr<E1, E2, Op>> {
   public:
    BinaryExpr(E1 e1, E2 e2, Op op) : e1_(e1), e2_(e2), op_(op) {}

    auto operator[](size_t i) const { return op_(e1_[i], e2_[i]); }

    size_t size() const { return std::min(e1_.size(), e2_.size()); }

   private:
    E1 e1_;
    E2 e2_;
    Op op_;
  };

  // 二元操作符重载
  template <typename E1, typename E2>
  auto operator+(const Expr<E1>& e1, const Expr<E2>& e2) const {
    return BinaryExpr<E1, E2, std::plus<T>>(e1, e2, std::plus<T>());
  }

  template <typename E1, typename E2>
  auto operator-(const Expr<E1>& e1, const Expr<E2>& e2) const {
    return BinaryExpr<E1, E2, std::minus<T>>(e1, e2, std::minus<T>());
  }

  template <typename E1, typename E2>
  auto operator*(const Expr<E1>& e1, const Expr<E2>& e2) const {
    return BinaryExpr<E1, E2, std::multiplies<T>>(e1, e2, std::multiplies<T>());
  }

  template <typename E1, typename E2>
  auto operator/(const Expr<E1>& e1, const Expr<E2>& e2) const {
    return BinaryExpr<E1, E2, std::divides<T>>(e1, e2, std::divides<T>());
  }

 private:
  AlignedAllocator<T> allocator_;
  T* data_;
  std::array<size_t, N> dimensions_;
  size_t total_elements_;
  size_t aligned_size_;
};

// 使用示例
int main() {
  MDVector<float, 2> A({3, 100});  // 3x100矩阵
  MDVector<float, 2> B({3, 100});
  MDVector<float, 2> C({3, 100});

  // 表达式模板运算
  C = A + B * 2.0f - A / 3.0f;

  // 高维张量
  MDVector<float, 3> D({37, 37, 300});
  MDVector<float, 3> E({37, 37, 300});
  MDVector<float, 3> F({37, 37, 300});

  // 张量运算
  F = D + E * 1.5f;

  return 0;
}