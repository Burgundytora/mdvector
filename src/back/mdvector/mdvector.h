#include "allocator.h"

// 表达式模板基类
template <typename Derived>
class Expr {
 public:
  auto operator[](size_t i) const { return static_cast<const Derived&>(*this)[i]; }
  size_t size() const { return static_cast<const Derived&>(*this).size(); }
};

// 核心MDVector类
template <typename T, size_t N>
class MDVector : public Expr<MDVector<T, N>> {
 public:
  // mdspan类型别名定义
  using extents_type = std::dextents<size_t, Dims>;
  using layout_type = std::layout_right;
  using mdspan_type = std::mdspan<T, extents_type, layout_type>;

 private:
  AlignedAllocator<T> allocator_;
  T* data_;
  mdspan_type view_;
  std::array<size_t, N> dimensions_;
  size_t total_elements_;
  size_t aligned_size_;

 public:
  // 构造函数
  template <typename... Dims>
  MDVector(Dims... dims) : dimensions_{static_cast<size_t>(dims)...} {
    static_assert(sizeof...(Dims) == N, "维度数量不匹配");
    total_elements_ = 1;
    for (auto d : dimensions_) total_elements_ *= d;
    aligned_size_ = allocator_.align_size(total_elements_);
    data_ = allocator_.allocate(aligned_size_);
    view_ = mdspan_type(data_, dimensions_);
  }

  // ========================================================
  // 访问运算符 提供safe 和 unsafe两种方式
  // (i, j, k) unsafe style
  template <typename... Indices>
  T& operator()(Indices... indices) {
    static_assert(sizeof...(Indices) == N, "索引数量与维度不匹配");
    return view_(static_cast<size_t>(indices)...);
  }
  // [i, j, k] unsafe style
  template <typename... Indices>
  T& operator[](Indices... indices) {
    static_assert(sizeof...(Indices) == N, "索引数量与维度不匹配");
    return view_(static_cast<size_t>(indices)...);
  }
  // .at(i, j, k) safe style
  template <typename... Indices>
  T& at(Indices... indices) {
    static_assert(sizeof...(Indices) == Dims, "Number of indices must match dimensions");
    size_t i = 0;
    for (auto len : array<size_t, Dims>{static_cast<size_t>(indices)...}) {
      if (len > view_.extent(i)) {
        std::cerr << "mdspan out-of-range error: " << len << ">" << view_.extent(i) << "\n";
        std::abort();
      }
      i++;
    }
    return view_[indices...];
  }

  // 运算符 通用接口
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
#ifdef MDVECTOR_USE_MKL
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
    for (size_t i = 0; i < total_elements_; i += SimdConfig<T>::mkl_vsize) {
      const size_t remain = std::min(SimdConfig<T>::mkl_vsize, total_elements_ - i);
      if constexpr (std::is_same_v<T, float>) {
        vsCopy(remain, &expr[i], data_ + i);
      } else {
        vdCopy(remain, &expr[i], data_ + i);
      }
    }
  }
};
