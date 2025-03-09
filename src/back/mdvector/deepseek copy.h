#include <memory>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <mkl.h>
#include <functional>

// 表达式模板基类
template <typename Derived>
struct Expr {
  const Derived& cast() const { return static_cast<const Derived&>(*this); }
  auto operator[](size_t i) const { return cast().operator[](i); }
  size_t size() const { return cast().size(); }
};

// SIMD 配置模板
template <typename T>
struct SimdConfig;
template <>
struct SimdConfig<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using simd_type = __m256;
  static constexpr MKL_INT mkl_vsize = 8;
};
template <>
struct SimdConfig<double> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 4;
  using simd_type = __m256d;
  static constexpr MKL_INT mkl_vsize = 4;
};

// 内存对齐分配器
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  T* allocate(size_t n) {
    size_t aligned_size = ((n + SimdConfig<T>::pack_size - 1) / SimdConfig<T>::pack_size) * SimdConfig<T>::pack_size;
    void* ptr = aligned_alloc(SimdConfig<T>::alignment, aligned_size * sizeof(T));
    return static_cast<T*>(ptr);
  }
  void deallocate(T* p, size_t) noexcept { free(p); }
};

// 核心MDVector类模板
template <typename T, size_t N = 1>
class MDVector : public Expr<MDVector<T, N>> {
 public:
  MDVector(std::initializer_list<size_t> dims) : dimensions_(dims) {
    total_elements_ = 1;
    for (auto dim : dimensions_) total_elements_ *= dim;
    data_ = allocator_.allocate(total_elements_);
  }

  template <typename... Args>
  T& operator()(Args... indices) {
    static_assert(sizeof...(indices) == N, "Incorrect number of indices");
    return data_[calculate_index(indices...)];
  }

  // AVX2优化实现
  MDVector& operator+=(const MDVector& rhs) {
    elementwise_op(rhs, [](auto a, auto b) { return a + b; });
    return *this;
  }
  MDVector& operator-=(const MDVector& rhs) {
    elementwise_op(rhs, [](auto a, auto b) { return a - b; });
    return *this;
  }
  MDVector& operator*=(const MDVector& rhs) {
    elementwise_op(rhs, [](auto a, auto b) { return a * b; });
    return *this;
  }
  MDVector& operator/=(const MDVector& rhs) {
    elementwise_op(rhs, [](auto a, auto b) { return a / b; });
    return *this;
  }

  // MKL优化入口
  void mkl_add(const MDVector& rhs) {
    mkl_elementwise(rhs, [](auto n, auto a, auto b, auto r) {
      if constexpr (std::is_same_v<T, float>)
        vsAdd(n, a, b, r);
      else
        vdAdd(n, a, b, r);
    });
  }

  void mkl_mul(const MDVector& rhs) {
    mkl_elementwise(rhs, [](auto n, auto a, auto b, auto r) {
      if constexpr (std::is_same_v<T, float>)
        vsMul(n, a, b, r);
      else
        vdMul(n, a, b, r);
    });
  }

  T* data() const { return data_; }
  size_t size() const { return total_elements_; }

  template <typename E>
  MDVector& operator=(const Expr<E>& expr) {
    const E& e = expr.cast();
#ifdef USE_MKL
    mkl_assign(e);
#else
    avx2_assign(e);
#endif
    return *this;
  }

 private:
  // AVX2通用元素操作
  template <typename F>
  void elementwise_op(const MDVector& rhs, F&& op) {
    constexpr size_t pack_size = SimdConfig<T>::pack_size;
    const size_t aligned_size = (total_elements_ / pack_size) * pack_size;

    // AVX2向量化部分
    if constexpr (std::is_same_v<T, float>) {
      for (size_t i = 0; i < aligned_size; i += pack_size) {
        auto a = _mm256_load_ps(data_ + i);
        auto b = _mm256_load_ps(rhs.data_ + i);
        _mm256_store_ps(data_ + i, op(a, b));
      }
    } else {
      for (size_t i = 0; i < aligned_size; i += pack_size) {
        auto a = _mm256_load_pd(data_ + i);
        auto b = _mm256_load_pd(rhs.data_ + i);
        _mm256_store_pd(data_ + i, op(a, b));
      }
    }

    // 标量处理尾部
    for (size_t i = aligned_size; i < total_elements_; ++i) {
      data_[i] = op(data_[i], rhs.data_[i]);
    }
  }

  // MKL通用元素操作
  template <typename F>
  void mkl_elementwise(const MDVector& rhs, F&& mkl_op) {
    mkl_op(total_elements_, data_, rhs.data_, data_);
  }

  // AVX2赋值实现
  template <typename E>
  void avx2_assign(const E& expr) {
    constexpr size_t pack_size = SimdConfig<T>::pack_size;
    const size_t aligned_size = (total_elements_ / pack_size) * pack_size;

    if constexpr (std::is_same_v<T, float>) {
      for (size_t i = 0; i < aligned_size; i += pack_size) {
        _mm256_store_ps(data_ + i, expr[i]);
      }
    } else {
      for (size_t i = 0; i < aligned_size; i += pack_size) {
        _mm256_store_pd(data_ + i, expr[i]);
      }
    }

    for (size_t i = aligned_size; i < total_elements_; ++i) {
      data_[i] = expr[i];
    }
  }

  // MKL赋值实现
  template <typename E>
  void mkl_assign(const E& expr) {
#pragma omp parallel for
    for (size_t i = 0; i < total_elements_; i += SimdConfig<T>::mkl_vsize) {
      const size_t remain = std::min(SimdConfig<T>::mkl_vsize, total_elements_ - i);
      if constexpr (std::is_same_v<T, float>) {
        vsCopy(remain, &expr[i], data_ + i);
      } else {
        vdCopy(remain, &expr[i], data_ + i);
      }
    }
  }

  // 其他成员保持不变...
};

// 二元运算表达式模板
template <typename L, typename R, typename Op>
struct BinaryExpr : Expr<BinaryExpr<L, R, Op>> {
  BinaryExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  auto operator[](size_t i) const { return Op::apply(lhs[i], rhs[i]); }
  size_t size() const { return lhs.size(); }

  const L& lhs;
  const R& rhs;
};

// 运算类型定义
struct OpAdd {
  template <typename T>
  static T apply(T a, T b) {
    return a + b;
  }
};
struct OpSub {
  template <typename T>
  static T apply(T a, T b) {
    return a - b;
  }
};
struct OpMul {
  template <typename T>
  static T apply(T a, T b) {
    return a * b;
  }
};
struct OpDiv {
  template <typename T>
  static T apply(T a, T b) {
    return a / b;
  }
};

// 运算符重载
template <typename L, typename R>
auto operator+(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, OpAdd>(l.cast(), r.cast());
}
template <typename L, typename R>
auto operator-(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, OpSub>(l.cast(), r.cast());
}
template <typename L, typename R>
auto operator*(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, OpMul>(l.cast(), r.cast());
}
template <typename L, typename R>
auto operator/(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, OpDiv>(l.cast(), r.cast());
}

// 使用示例
int main() {
  // 创建并初始化两个3维数组
  MDVector<float, 3> A({2, 3, 4}), B({2, 3, 4});
  // ...初始化代码...

  // 表达式模板计算（自动选择AVX2/MKL）
  MDVector<float, 3> C = A + B * (A - B);

  // 直接使用MKL优化
  MDVector<double> X(1024), Y(1024);
  X.mkl_add(Y);  // MKL加速加法
  Y.mkl_mul(X);  // MKL加速乘法

  return 0;
}