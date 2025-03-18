#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <iomanip>

// ======================== SIMD配置 ========================
template <typename T>
struct SimdConfig;
template <>
struct SimdConfig<float> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 8;
  using simd_type = __m256;
};
template <>
struct SimdConfig<double> {
  static constexpr size_t alignment = 32;
  static constexpr size_t pack_size = 4;
  using simd_type = __m256d;
};

// ======================== 内存分配器 ========================
template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  static T* allocate(size_t n) {
    void* ptr =
        aligned_alloc(SimdConfig<T>::alignment, ((n + SimdConfig<T>::pack_size - 1) / SimdConfig<T>::pack_size) *
                                                    SimdConfig<T>::pack_size * sizeof(T));
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  static void deallocate(T* p, size_t = 0) { free(p); }
};

// ======================== 表达式模板基类 ========================
template <typename Derived>
class Expr {
 public:
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  size_t size() const { return derived().size(); }

  template <typename Dest>
  void eval_to(Dest* dest) const {
    const size_t n = size();
    constexpr size_t pack_size = SimdConfig<std::remove_const_t<Dest>>::pack_size;
    size_t i = 0;

    for (; i <= n - pack_size; i += pack_size) {
      auto simd_val = derived().template eval_simd<std::remove_const_t<Dest>>(i);
      if constexpr (std::is_same_v<std::remove_const_t<Dest>, float>) {
        _mm256_store_ps(dest + i, simd_val);
      } else {
        _mm256_store_pd(dest + i, simd_val);
      }
    }

    // 处理尾部元素
    if (i < n) {
      alignas(SimdConfig<std::remove_const_t<Dest>>::alignment) std::remove_const_t<Dest> temp[pack_size] = {0};
      auto simd_val = derived().template eval_simd<std::remove_const_t<Dest>>(i);
      if constexpr (std::is_same_v<std::remove_const_t<Dest>, float>) {
        _mm256_store_ps(temp, simd_val);
      } else {
        _mm256_store_pd(temp, simd_val);
      }
      std::memcpy(dest + i, temp, (n - i) * sizeof(std::remove_const_t<Dest>));
    }
  }
};

// ======================== AVX2计算函数 ========================
template <typename T>
typename SimdConfig<T>::simd_type avx2_load(const T* ptr) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_load_ps(ptr);
  } else {
    return _mm256_load_pd(ptr);
  }
}

// ======================== 表达式类 ========================
template <typename L, typename R>
class AddExpr : public Expr<AddExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  AddExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_add_ps(l, r);
    } else {
      return _mm256_add_pd(l, r);
    }
  }
};

template <typename L, typename R>
class SubExpr : public Expr<SubExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  SubExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_sub_ps(l, r);
    } else {
      return _mm256_sub_pd(l, r);
    }
  }
};

template <typename L, typename R>
class MulExpr : public Expr<MulExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  MulExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  size_t size() const { return lhs.size(); }

  template <typename T>
  typename SimdConfig<T>::simd_type eval_simd(size_t i) const {
    auto l = lhs.template eval_simd<T>(i);
    auto r = rhs.template eval_simd<T>(i);
    if constexpr (std::is_same_v<T, float>) {
      return _mm256_mul_ps(l, r);
    } else {
      return _mm256_mul_pd(l, r);
    }
  }
};

// ======================== 运算符重载 ========================
template <typename L, typename R>
AddExpr<L, R> operator+(const Expr<L>& lhs, const Expr<R>& rhs) {
  return AddExpr<L, R>(lhs.derived(), rhs.derived());
}

template <typename L, typename R>
SubExpr<L, R> operator-(const Expr<L>& lhs, const Expr<R>& rhs) {
  return SubExpr<L, R>(lhs.derived(), rhs.derived());
}

template <typename L, typename R>
MulExpr<L, R> operator*(const Expr<L>& lhs, const Expr<R>& rhs) {
  return MulExpr<L, R>(lhs.derived(), rhs.derived());
}

// ======================== MDVector类 ========================
template <typename T, size_t Dims>
class MDVector : public Expr<MDVector<T, Dims>> {
  std::vector<T, AlignedAllocator<T>> data_;
  size_t total_elements_;

 public:
  explicit MDVector(size_t size) : total_elements_(size) { data_.resize(size); }

  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }
  size_t size() const { return total_elements_; }

  template <typename E>
  MDVector& operator=(const Expr<E>& expr) {
    expr.eval_to(data_.data());
    return *this;
  }

  template <typename T2>
  typename SimdConfig<T2>::simd_type eval_simd(size_t i) const {
    return avx2_load<T2>(data_.data() + i);
  }

  void set_value(T val) { std::fill(data_.begin(), data_.end(), val); }
};

// ======================== 性能测试 ========================
void benchmark() {
  const size_t sizes[] = {10, 100, 1000, 10000, 100000};
  const int trials = 10000;

  for (auto size : sizes) {
    MDVector<float, 1> a(size), b(size), c(size), d(size), e(size);
    a.set_value(1.0f);
    b.set_value(2.0f);
    c.set_value(3.0f);
    d.set_value(4.0f);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
      e = a + b * c - d;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t2 - t1).count() * 1e6 / trials;

    for (size_t i = 0; i < size; ++i) {
      assert(fabs(e.data()[i] - (1.0f + 2.0f * 3.0f - 4.0f)) < 1e-6);
    }

    std::cout << "Size: " << std::setw(7) << size << " | Time: " << std::setw(6) << std::fixed << std::setprecision(2)
              << elapsed << " μs\n";
  }
}
