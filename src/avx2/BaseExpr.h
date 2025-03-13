// ========================================================
// 表达式模板
template <typename Derived>
class Expr {
 public:
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  // 关键方法：直接操作目标内存
  template <typename T>
  void eval_to(T* __restrict dest) const {
    derived().eval_to_impl(dest);
  }
};

// 标量表达式模板
template <typename T>
class ScalarExpr : public Expr<ScalarExpr<T>> {
  T value_;

 public:
  explicit ScalarExpr(T value) : value_(value) {}

  template <typename U>
  void eval_to_impl(U* dest) const {
    const size_t n = /* 获取目标数组大小 */;
    avx2_set_scalar(value_, dest, n);
  }

  T value() const { return value_; }
};

// 标量设置函数
template <typename T>
void avx2_set_scalar(T scalar, T* dest, size_t n) {
  constexpr size_t pack_size = SimdConfig<T>::pack_size;
  typename SimdConfig<T>::simd_type scalar_vec;
  if constexpr (std::is_same_v<T, float>) {
    scalar_vec = _mm256_set1_ps(scalar);
  } else {
    scalar_vec = _mm256_set1_pd(scalar);
  }

  size_t i = 0;
  for (; i <= n - pack_size; i += pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      _mm256_store_ps(dest + i, scalar_vec);
    } else {
      _mm256_store_pd(dest + i, scalar_vec);
    }
  }
  // 处理尾部元素...
}