#ifndef HEADER_AVX2_H_
#define HEADER_AVX2_H_

#include "allocator.h"

// 1.size小于10000时 显著快于for循环、eigen、mkl
// 2.size在10左右时double 20左右float 有概率很慢
// 3.特大size进行缓存流水线优化

// ========================================================
// 基础运算赋值 c = a ? b
// c = a + b
template <class T>
void avx2_add(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_add_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_add_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// c = a - b
template <class T>
void avx2_sub(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_sub_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_sub_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// c = a * b
template <class T>
void avx2_mul(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_mul_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_mul_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// c = a / b
template <class T>
void avx2_div(const T* a, const T* b, T* c, size_t n) {
  // float double int
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "type must be float or double!");
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // float
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_div_ps(va, vb);
      _mm256_store_ps(c + i, vc);
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      // double
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_div_pd(va, vb);
      _mm256_store_pd(c + i, vc);
    }
  }
}

// ========================================================
// 基础累积运算 c ?= a
// c += a
template <class T>
void avx2_add_inplace(const T* a, T* c, size_t n) {
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_add_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_add_pd(vc, va));
    }
  }
}

// c -= a
template <class T>
void avx2_sub_inplace(const T* a, T* c, size_t n) {
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_sub_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_sub_pd(vc, va));
    }
  }
}

// c *= a
template <class T>
void avx2_mul_inplace(const T* a, T* c, size_t n) {
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_mul_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_mul_pd(vc, va));
    }
  }
}

// c /= a
template <class T>
void avx2_div_inplace(const T* a, T* c, size_t n) {
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vc = _mm256_load_ps(c + i);
      _mm256_store_ps(c + i, _mm256_div_ps(vc, va));
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vc = _mm256_load_pd(c + i);
      _mm256_store_pd(c + i, _mm256_div_pd(vc, va));
    }
  }
}

// // ========================================================
// //  复杂指令 c ?= a ? b
// // 是否用得到？ 针对性优化
// // 尾部掩码处理 相比幽灵元素如何？
// //  c += a + b
// template <class T>
// void avx2_add_accumulate(const T* a, const T* b, T* c, size_t n) {
//   constexpr size_t pack_size = SimdConfig<T>::pack_size;
//   size_t i = 0;

//   // 主循环（展开两次减少分支预测开销）
//   for (; i + 2 * pack_size <= n; i += 2 * pack_size) {
//     if constexpr (std::is_same_v<T, float>) {
//       // 第一组
//       __m256 va0 = _mm256_load_ps(a + i);
//       __m256 vb0 = _mm256_load_ps(b + i);
//       __m256 sum0 = _mm256_add_ps(va0, vb0);
//       __m256 vc0 = _mm256_load_ps(c + i);
//       _mm256_store_ps(c + i, _mm256_add_ps(vc0, sum0));

//       // 第二组
//       __m256 va1 = _mm256_load_ps(a + i + pack_size);
//       __m256 vb1 = _mm256_load_ps(b + i + pack_size);
//       __m256 sum1 = _mm256_add_ps(va1, vb1);
//       __m256 vc1 = _mm256_load_ps(c + i + pack_size);
//       _mm256_store_ps(c + i + pack_size, _mm256_add_ps(vc1, sum1));
//     } else {
//       // 类似逻辑处理double
//       __m256d va0 = _mm256_load_pd(a + i);
//       __m256d vb0 = _mm256_load_pd(b + i);
//       __m256d sum0 = _mm256_add_pd(va0, vb0);
//       __m256d vc0 = _mm256_load_pd(c + i);
//       _mm256_store_pd(c + i, _mm256_add_pd(vc0, sum0));

//       __m256d va1 = _mm256_load_pd(a + i + pack_size);
//       __m256d vb1 = _mm256_load_pd(b + i + pack_size);
//       __m256d sum1 = _mm256_add_pd(va1, vb1);
//       __m256d vc1 = _mm256_load_pd(c + i + pack_size);
//       _mm256_store_pd(c + i + pack_size, _mm256_add_pd(vc1, sum1));
//     }
//   }

//   // 处理剩余完整块
//   for (; i <= n - pack_size; i += pack_size) {
//     if constexpr (std::is_same_v<T, float>) {
//       __m256 va = _mm256_load_ps(a + i);
//       __m256 vb = _mm256_load_ps(b + i);
//       __m256 sum = _mm256_add_ps(va, vb);
//       __m256 vc = _mm256_load_ps(c + i);
//       _mm256_store_ps(c + i, _mm256_add_ps(vc, sum));
//     } else {
//       __m256d va = _mm256_load_pd(a + i);
//       __m256d vb = _mm256_load_pd(b + i);
//       __m256d sum = _mm256_add_pd(va, vb);
//       __m256d vc = _mm256_load_pd(c + i);
//       _mm256_store_pd(c + i, _mm256_add_pd(vc, sum));
//     }
//   }

//   // 尾部处理（掩码优化）
//   size_t remaining = n - i;
//   if (remaining > 0) {
//     if constexpr (std::is_same_v<T, float>) {
//       __m256i mask = _mm256_setr_epi32((remaining > 0) ? -1 : 0, (remaining > 1) ? -1 : 0, (remaining > 2) ? -1 : 0,
//                                        (remaining > 3) ? -1 : 0, (remaining > 4) ? -1 : 0, (remaining > 5) ? -1 : 0,
//                                        (remaining > 6) ? -1 : 0, (remaining > 7) ? -1 : 0);
//       __m256 va = _mm256_maskload_ps(a + i, mask);
//       __m256 vb = _mm256_maskload_ps(b + i, mask);
//       __m256 sum = _mm256_add_ps(va, vb);
//       __m256 vc = _mm256_maskload_ps(c + i, mask);
//       _mm256_maskstore_ps(c + i, mask, _mm256_add_ps(vc, sum));
//     } else {
//       __m256i mask = _mm256_setr_epi64x((remaining > 0) ? -1 : 0, (remaining > 1) ? -1 : 0, (remaining > 2) ? -1 : 0,
//                                         (remaining > 3) ? -1 : 0);
//       __m256d va = _mm256_maskload_pd(a + i, mask);
//       __m256d vb = _mm256_maskload_pd(b + i, mask);
//       __m256d sum = _mm256_add_pd(va, vb);
//       __m256d vc = _mm256_maskload_pd(c + i, mask);
//       _mm256_maskstore_pd(c + i, mask, _mm256_add_pd(vc, sum));
//     }
//   }
// }
// ========================================================

// FMA操作
// d[i] = a[i] * b[i] + c[i]
template <class T>
void avx2_fma_d_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fmadd_ps(va, vb, vc);
      _mm256_store_ps(d + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fmadd_pd(va, vb, vc);
      _mm256_store_pd(d + i, result);
    }
  }
}

// c[i] = a[i] * b[i] + c[i]
template <class T>
void avx2_fma_c_abc(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fmadd_ps(va, vb, vc);
      _mm256_store_ps(c + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fmadd_pd(va, vb, vc);
      _mm256_store_pd(c + i, result);
    }
  }
}

// d[i] = a[i] * b[i] - c[i]
template <class T>
void avx2_fms_d_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fmsub_ps(va, vb, vc);
      _mm256_store_ps(d + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fmsub_pd(va, vb, vc);
      _mm256_store_pd(d + i, result);
    }
  }
}

// c[i] = a[i] * b[i] - c[i]
template <class T>
void avx2_fms_c_abc(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fmsub_ps(va, vb, vc);
      _mm256_store_ps(c + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fmsub_pd(va, vb, vc);
      _mm256_store_pd(c + i, result);
    }
  }
}

// d[i] = -(a[i] * b[i]) + c[i]
template <class T>
void avx2_fnma_d_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fnmadd_ps(va, vb, vc);
      _mm256_store_ps(d + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fnmadd_pd(va, vb, vc);
      _mm256_store_pd(d + i, result);
    }
  }
}

// c[i] = -(a[i] * b[i]) + c[i]
template <class T>
void avx2_fnma_c_abc(const T* a, const T* b, T* c, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fnmadd_ps(va, vb, vc);
      _mm256_store_ps(c + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fnmadd_pd(va, vb, vc);
      _mm256_store_pd(c + i, result);
    }
  }
}

// d[i] = -(a[i] * b[i]) + c[i]
template <class T>
void avx2_fnms_d_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fnmsub_ps(va, vb, vc);
      _mm256_store_ps(d + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fnmsub_pd(va, vb, vc);
      _mm256_store_pd(d + i, result);
    }
  }
}

// c[i] = -(a[i] * b[i]) + c[i]
template <class T>
void avx2_fnms_c_abc(const T* a, const T* b, T* c, T* d, size_t n) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Type must be float or double!");
  for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
    if constexpr (std::is_same_v<T, float>) {
      __m256 va = _mm256_load_ps(a + i);
      __m256 vb = _mm256_load_ps(b + i);
      __m256 vc = _mm256_load_ps(c + i);
      __m256 result = _mm256_fnmsub_ps(va, vb, vc);
      _mm256_store_ps(c + i, result);
    } else {
      __m256d va = _mm256_load_pd(a + i);
      __m256d vb = _mm256_load_pd(b + i);
      __m256d vc = _mm256_load_pd(c + i);
      __m256d result = _mm256_fnmsub_pd(va, vb, vc);
      _mm256_store_pd(c + i, result);
    }
  }
}

// 辅助AVX2复制函数
template <class T>
void avx2_copy(const T* src, T* dest, size_t n) {
  if constexpr (std::is_same_v<T, float>) {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      _mm256_store_ps(dest + i, _mm256_load_ps(src + i));
    }
  } else {
    for (size_t i = 0; i < n; i += SimdConfig<T>::pack_size) {
      _mm256_store_pd(dest + i, _mm256_load_pd(src + i));
    }
  }
}
// ... 其他融合指令

// 表达式模板AVX2版本
template <typename Derived>
class VectorExpr {
 public:
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  // 关键方法：直接操作目标内存
  template <typename T>
  void eval_to(T* __restrict dest) const {
    derived().eval_to_impl(dest);
  }
};

template <typename L, typename R>
class AddExpr : public VectorExpr<AddExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  AddExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  template <class T>
  void eval_to_impl(T* __restrict dest) const {
    avx2_add(lhs.data(), rhs.data(), dest, lhs.size());
  }
};

template <typename L, typename R>
class SubExpr : public VectorExpr<SubExpr<L, R>> {
  const L& lhs;
  const R& rhs;

 public:
  SubExpr(const L& l, const R& r) : lhs(l), rhs(r) {}

  template <class T>
  void eval_to_impl(T* __restrict dest) const {
    avx2_sub(lhs.data(), rhs.data(), dest, lhs.size());
  }
};

// 运算符重载返回表达式
template <typename L, typename R>
AddExpr<L, R> operator+(const VectorExpr<L>& lhs, const VectorExpr<R>& rhs) {
  return AddExpr<L, R>(lhs.derived(), rhs.derived());
}

template <typename L, typename R>
SubExpr<L, R> operator-(const VectorExpr<L>& lhs, const VectorExpr<R>& rhs) {
  return SubExpr<L, R>(lhs.derived(), rhs.derived());
}

#endif  // HEADER_AVX2_H_